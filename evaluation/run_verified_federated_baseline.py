"""Verified 5-seed FL-LR study; all reported artifacts originate here."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, matthews_corrcoef,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, auc, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'verified'
OUT.mkdir(parents=True,exist_ok=True)
INSDN=Path(os.environ.get('INSDN_CSV',ROOT/'data'/'insdn'/'flow_stats.csv'))
# Three views of the same target traffic. `trafficlabelling` keeps the original
# Protocol and Source Port columns. `tl_counterfactual` is the controlled
# comparison: the *same rows* as `trafficlabelling`, with every other feature and
# the label untouched, and only those two fields replaced by the surrogates that
# MachineLearningCVE forces. `mlcve` is the actual public release, which differs
# from the other two in row set and attack prevalence as well, so it establishes
# practical relevance rather than the mechanism.
MIRRORS={'trafficlabelling':Path(os.environ.get('CICIDS2017_MIRROR_CSV',ROOT/'data'/'cicids2017_mirror'/'test.csv')),
         'tl_counterfactual':Path(os.environ.get('CICIDS2017_COUNTERFACTUAL_CSV',ROOT/'data'/'cicids2017_mirror_counterfactual'/'test.csv')),
         'mlcve':Path(os.environ.get('CICIDS2017_MLCVE_MIRROR_CSV',ROOT/'data'/'cicids2017_mirror_mlcve'/'test.csv'))}
CROSS_KEYS=[f'cross_{m}' for m in MIRRORS]
CENTRAL_CROSS_KEYS=[f'central_cross_{m}' for m in MIRRORS]
PROVENANCE_FIELDS=('ip_proto','tp_src')
# The prepared InSDN and mirror CSVs also carry a `flow_duration` column that is
# byte-identical to `duration_sec` in every row; it is excluded here so that the
# model sees nine independent inputs rather than one duplicated coefficient.
FEATURES=['ip_proto','tp_src','tp_dst','packet_count','byte_count','duration_sec','packet_count_per_sec','byte_count_per_sec','packet_size_avg']
SEEDS=[11,23,42,77,101]; K=5; ROUNDS=20; N=30000

def load(path,label,kind,seed):
 d=pd.read_csv(path,usecols=FEATURES+[label]); raw=d[label].astype(str).str.lower().str.strip()
 y=(~raw.isin(['normal','benign','0','0.0'])).astype(int).to_numpy() if kind=='insdn' else (~raw.isin(['1','normal','benign'])).astype(int).to_numpy()
 x=d[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy(float)
 if len(x)>N: _,x,_,y=train_test_split(x,y,test_size=N,random_state=seed,stratify=y)
 return x,y
def clf(seed): return SGDClassifier(loss='log_loss',penalty='l2',alpha=1e-4,learning_rate='constant',eta0=0.01,max_iter=1,tol=None,random_state=seed,shuffle=True)
def initialized_clf(seed,n_features):
 """Create a valid sklearn state and reset it to a client-neutral zero model."""
 m=clf(seed);m.partial_fit(np.zeros((2,n_features)),np.array([0,1]),classes=np.array([0,1]))
 m.coef_.fill(0.);m.intercept_.fill(0.)
 return m
def fit(x,y,seed,epochs=ROUNDS):
 m=clf(seed)
 for _ in range(epochs): m.partial_fit(x,y,classes=np.array([0,1]))
 return m
def prob(coef,bias,x):
 z=np.clip(x@coef.ravel()+float(bias[0]),-35,35); return 1/(1+np.exp(-z))
def met(y,p):
 q=(p>=.5).astype(int); pr,re,f,_=precision_recall_fscore_support(y,q,average='binary',zero_division=0); fpr,tpr,_=roc_curve(y,p); pc,rc,_=precision_recall_curve(y,p)
 tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
 return dict(precision=float(pr),recall=float(re),f1=float(f),roc_auc=float(roc_auc_score(y,p)),pr_auc=float(auc(rc,pc)),balanced_accuracy=float(balanced_accuracy_score(y,q)),mcc=float(matthews_corrcoef(y,q)),accuracy=float(accuracy_score(y,q)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),fpr=fpr,tpr=tpr,pc=pc,rc=rc)
def parts(x,y,seed,non_iid=False):
 rng=np.random.default_rng(seed)
 if not non_iid:
  ids=np.arange(len(y));rng.shuffle(ids); return [(x[a],y[a]) for a in np.array_split(ids,K)]
 bins=[[] for _ in range(K)]
 for c in [0,1]:
  ids=np.flatnonzero(y==c);rng.shuffle(ids); cuts=(np.cumsum(rng.dirichlet(np.ones(K)*.5))[:-1]*len(ids)).astype(int)
  for b,a in zip(bins,np.split(ids,cuts)): b.extend(a.tolist())
 out=[]
 for b in bins:
  b=np.array(b);rng.shuffle(b);out.append((x[b],y[b]))
 return out
def fed(ps,xt,yt,seed):
 n_features=ps[0][0].shape[1];w=np.zeros((1,n_features));b=np.zeros(1);hist=[]
 for r in range(ROUNDS):
  updates=[]
  for i,(x,y) in enumerate(ps):
   m=initialized_clf(seed*100+i,n_features);m.coef_=w.copy();m.intercept_=b.copy();m.partial_fit(x,y);updates.append((m.coef_,m.intercept_,len(y)))
  a=np.array([u[2] for u in updates],float);a/=a.sum();w=sum(z*u[0] for z,u in zip(a,updates));b=sum(z*u[1] for z,u in zip(a,updates));hist.append(met(yt,prob(w,b,xt))['f1'])
 return w,b,hist
def ks_distance(a,b):
 """Two-sample empirical KS distance without an additional SciPy dependency."""
 a=np.sort(np.asarray(a));b=np.sort(np.asarray(b));grid=np.sort(np.concatenate([a,b]))
 return float(np.max(np.abs(np.searchsorted(a,grid,side='right')/len(a)-np.searchsorted(b,grid,side='right')/len(b))))
def compact(d): return {k:v for k,v in d.items() if isinstance(v,(float,int,np.floating,np.integer))}
def prov_checks(s,t):
 """Label-free provenance checks: source-train stats vs unlabelled target values.

 Two checks form the proposed screen. `degenerate` fires when a target column has
 lost essentially all of its variance, and `support_violation` fires when target
 values fall outside the range ever observed in source training. Both detect a
 placeholder substituted for an absent column. `tv_dist` and `collapsed_category`
 are additional candidates that we report but do not adopt, because Section V
 shows they do not separate a harmful reconstruction from a benign one.
 """
 ss,ts=float(s.std()),float(t.std());lo,hi=s.min(),s.max()
 r=dict(std_ratio=ts/ss if ss else float('nan'),out_of_range=float(np.mean((t<lo)|(t>hi))))
 u=np.unique(s);r['discrete']=bool(len(u)<=16)
 if r['discrete']:
  sp={v:float(np.mean(s==v)) for v in u};tp={v:float(np.mean(t==v)) for v in np.unique(t)}
  r['tv_dist']=float(sum(abs(sp.get(k,0.)-tp.get(k,0.)) for k in set(sp)|set(tp))/2)
  r['collapsed_category']=bool(any(p>.05 and tp.get(v,0.)<.01 for v,p in sp.items()))
 else: r['tv_dist']=float('nan');r['collapsed_category']=False
 r['degenerate']=bool(r['std_ratio']<.01);r['support_violation']=bool(r['out_of_range']>.01)
 r['flagged']=bool(r['degenerate'] or r['support_violation'])
 return r
def one(seed,ablation='all'):
 xi,yi=load(INSDN,'label','insdn',seed); tg={m:load(p,'label','cic',seed) for m,p in MIRRORS.items()}
 # The controlled claim depends on the counterfactual sample still being the same
 # rows after seeded subsampling, so assert it instead of trusting it.
 if {'trafficlabelling','tl_counterfactual'}<=tg.keys():
  ua=[i for i,f in enumerate(FEATURES) if f not in PROVENANCE_FIELDS]
  (xa,ya),(xb,yb)=tg['trafficlabelling'],tg['tl_counterfactual']
  if not (np.array_equal(xa[:,ua],xb[:,ua]) and np.array_equal(ya,yb)):
   raise SystemExit('counterfactual mirror is not row-matched after seeded subsampling')
  if np.array_equal(xa[:,[FEATURES.index(f) for f in PROVENANCE_FIELDS]],xb[:,[FEATURES.index(f) for f in PROVENANCE_FIELDS]]):
   raise SystemExit('counterfactual mirror is identical to the original; nothing was varied')
 # 'intersection' keeps only the fields MachineLearningCVE supplies as read
 # values, i.e. it drops both provenance-sensitive columns instead of
 # reconstructing or sentinel-filling them.
 keep={'no_ports':[0,3,4,5,6,7,8],'no_protocol':list(range(1,len(FEATURES))),
       'intersection':[i for i,f in enumerate(FEATURES) if f not in PROVENANCE_FIELDS]}.get(ablation)
 if keep is not None: xi=xi[:,keep];tg={m:(x[:,keep],y) for m,(x,y) in tg.items()}
 xtr,xte,ytr,yte=train_test_split(xi,yi,test_size=.2,stratify=yi,random_state=seed)
 raw_tr=xtr.copy();raw_tg={m:x.copy() for m,(x,_) in tg.items()}  # unscaled, for the label-free screen
 if ablation=='onehot_protocol':
  enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(xtr[:,[0]])
  hot=lambda a:np.hstack([a[:,1:],enc.transform(a[:,[0]])])
  xtr=hot(xtr); xte=hot(xte); tg={m:(hot(x),y) for m,(x,y) in tg.items()}
 sc=StandardScaler().fit(xtr);xtr,xte=sc.transform(xtr),sc.transform(xte);tg={m:(sc.transform(x),y) for m,(x,y) in tg.items()}
 cm=fit(xtr,ytr,seed); cen=met(yte,cm.predict_proba(xte)[:,1]); pi=parts(xtr,ytr,seed);pn=parts(xtr,ytr,seed,True)
 wi,bi,hi=fed(pi,xte,yte,seed);wn,bn,hn=fed(pn,xte,yte,seed); fi=met(yte,prob(wi,bi,xte));fn=met(yte,prob(wn,bn,xte))
 cross={m:met(y,prob(wi,bi,x)) for m,(x,y) in tg.items()}
 # Scored so that the provenance effect cannot be attributed to FedAvg averaging.
 central_cross={m:met(y,cm.predict_proba(x)[:,1]) for m,(x,y) in tg.items()}
 local=[met(yte,fit(x,y,seed+i).predict_proba(xte)[:,1]) for i,(x,y) in enumerate(pi)]
 audit=[];screen=[]
 if ablation=='all':
  for m in tg:
   for j,name in enumerate(FEATURES):
    screen.append(dict(mirror=m,feature=name,**prov_checks(raw_tr[:,j],raw_tg[m][:,j])))
  for m,(xc,yc) in tg.items():
   for j,name in enumerate(FEATURES):
    sx=xte.copy();tx=xc.copy();sx[:,j]=0.;tx[:,j]=0.
    sm=met(yte,prob(wi,bi,sx));tm=met(yc,prob(wi,bi,tx))
    audit.append(dict(mirror=m,feature=name,ks=ks_distance(xtr[:,j],xc[:,j]),target_abs_z_mean=float(abs(xc[:,j].mean())),source_mask_f1=float(sm['f1']),mirror_mask_roc_auc=float(tm['roc_auc']),mirror_mask_mcc=float(tm['mcc']),mirror_auc_delta=float(tm['roc_auc']-cross[m]['roc_auc'])))
 out=dict(seed=seed,centralized=compact(cen),local_mean={k:float(np.mean([a[k] for a in local])) for k in ['f1','roc_auc','pr_auc','balanced_accuracy','mcc','recall']},fedavg_iid=compact(fi),fedavg_noniid=compact(fn),feature_audit=audit,provenance_screen=screen,iid_clients=[dict(n=len(y),benign=int((y==0).sum()),attack=int((y==1).sum())) for _,y in pi],noniid_clients=[dict(n=len(y),benign=int((y==0).sum()),attack=int((y==1).sum())) for _,y in pn],history_iid=hi,history_noniid=hn,counts=dict(insdn_benign=int((yi==0).sum()),insdn_attack=int((yi==1).sum())))
 for m,(x,y) in tg.items():
  out[f'cross_{m}']=compact(cross[m]);out[f'central_cross_{m}']=compact(central_cross[m]);out['counts'][f'{m}_benign']=int((y==0).sum());out['counts'][f'{m}_attack']=int((y==1).sum())
 return out
def summarize(rows,key):
 keys=['f1','roc_auc','pr_auc','balanced_accuracy','mcc','recall'];return {k:{'mean':float(np.mean([r[key][k] for r in rows])),'std':float(np.std([r[key][k] for r in rows],ddof=1))} for k in keys}
def main():
 missing=[str(p) for p in [INSDN,*MIRRORS.values()] if not p.exists()]
 if missing: raise SystemExit('Missing prepared dataset file(s): '+', '.join(missing)+'\nSee data/README.md or set INSDN_CSV, CICIDS2017_MIRROR_CSV and CICIDS2017_MLCVE_MIRROR_CSV.')
 rows=[one(s) for s in SEEDS]; ab=[one(s,'no_ports') for s in SEEDS]; npv=[one(s,'no_protocol') for s in SEEDS]; oh=[one(s,'onehot_protocol') for s in SEEDS]; isec=[one(s,'intersection') for s in SEEDS]
 groups=[('all_nine',rows),('without_ports',ab),('without_protocol',npv),('onehot_protocol',oh),('intersection',isec)]
 out=dict(seeds=SEEDS,clients=K,rounds=ROUNDS,features=FEATURES,source=dict(insdn_file=INSDN.name,mirrors={m:str(p.parent.name+'/'+p.name) for m,p in MIRRORS.items()},cicids2017_label_mapping={'1':'benign','0':'attack_ddos','2':'attack_portscan'},provenance='See data/README.md and evaluation/prepare_public_data.py'),hyperparameters=dict(model='SGDClassifier(log_loss)',penalty='l2',alpha=1e-4,eta0=.01,batch='full local partition per round',local_epochs=1,initialization='shared zero coefficients/intercept',threshold=.5,class_weight=None,scaler='StandardScaler fit on source training only',sklearn=sklearn.__version__),per_seed=rows,summary={k:summarize(rows,k) for k in ['centralized','local_mean','fedavg_iid','fedavg_noniid',*CROSS_KEYS,*CENTRAL_CROSS_KEYS]})
 for name,group in groups[1:]:
  out[f'ablation_{name}']=summarize(group,'fedavg_iid')
  for ck in CROSS_KEYS: out[f'ablation_{name}_{ck}']=summarize(group,ck)
 (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf8')
 pd.DataFrame([{**{'seed':r['seed']},**{f'{m}_{q}':r[m][q] for m in ['centralized','local_mean','fedavg_iid','fedavg_noniid',*CROSS_KEYS,*CENTRAL_CROSS_KEYS] for q in ['f1','roc_auc','pr_auc','balanced_accuracy','mcc']}} for r in rows]).to_csv(OUT/'five_seed_metrics.csv',index=False)
 pd.DataFrame([{'seed':z['seed'],'mirror':m,'setting':name,**{q:z[f'cross_{m}'][q] for q in ['f1','roc_auc','pr_auc','balanced_accuracy','mcc']}} for name,group in groups for z in group for m in MIRRORS]).to_csv(OUT/'external_ablation_metrics.csv',index=False)
 pd.DataFrame([{'seed':r['seed'],**a} for r in rows for a in r['feature_audit']]).to_csv(OUT/'feature_shift_audit.csv',index=False)
 pd.DataFrame([{'seed':r['seed'],**a} for r in rows for a in r['provenance_screen']]).to_csv(OUT/'provenance_screen.csv',index=False)
 r=rows[2];pd.DataFrame({'round':range(1,ROUNDS+1),'iid_f1':r['history_iid'],'noniid_f1':r['history_noniid']}).to_csv(OUT/'fedavg_history_seed42.csv',index=False)
 print(json.dumps(out['summary'],indent=2))
if __name__=='__main__': main()
