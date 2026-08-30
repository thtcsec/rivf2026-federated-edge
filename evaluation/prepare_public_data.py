"""Build the two prepared CSVs used by the RIVF 2026 audit.

Third-party data are not redistributed.  Supply the binary InSDN Dataset.csv
and the three named CICIDS2017 CSV/Parquet files documented in data/README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURES=['ip_proto','tp_src','tp_dst','packet_count','byte_count','duration_sec','packet_count_per_sec','byte_count_per_sec','packet_size_avg','flow_duration']
CIC_FILES={
 'monday':'Monday-WorkingHours.pcap_ISCX.csv',
 'portscan':'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
 'ddos':'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
}
LABEL_MAP={'BENIGN':'normal','PortScan':'portscan','DDoS':'ddos'}
TCP_FLAGS=['FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','ECE Flag Count','CWE Flag Count']

def read_frame(path:Path)->pd.DataFrame:
 if path.suffix.lower()=='.parquet': return pd.read_parquet(path)
 return pd.read_csv(path)

def locate(root:Path,name:str)->Path:
 # Prefer the validated Parquet mirrors when both a placeholder CSV and a
 # Parquet file are present.
 candidates=[root/(name+'.parquet'),root/name]
 for path in candidates:
  if path.exists(): return path
 raise FileNotFoundError(f'Missing {name} (CSV) or {name}.parquet under {root}')

def prepare_insdn(src:Path,out:Path)->None:
 cols=['Src Port','Dst Port','Protocol','Flow Duration','Tot Fwd Pkts','Tot Bwd Pkts','TotLen Fwd Pkts','TotLen Bwd Pkts','Flow Byts/s','Flow Pkts/s','Pkt Size Avg','target']
 d=pd.read_csv(src,usecols=cols).replace([np.inf,-np.inf],np.nan)
 z=pd.DataFrame({
  'ip_proto':pd.to_numeric(d['Protocol'],errors='coerce'),'tp_src':pd.to_numeric(d['Src Port'],errors='coerce'),'tp_dst':pd.to_numeric(d['Dst Port'],errors='coerce'),
  'packet_count':pd.to_numeric(d['Tot Fwd Pkts'],errors='coerce').fillna(0)+pd.to_numeric(d['Tot Bwd Pkts'],errors='coerce').fillna(0),
  'byte_count':pd.to_numeric(d['TotLen Fwd Pkts'],errors='coerce').fillna(0)+pd.to_numeric(d['TotLen Bwd Pkts'],errors='coerce').fillna(0),
  'duration_sec':pd.to_numeric(d['Flow Duration'],errors='coerce')/1e6,'packet_count_per_sec':pd.to_numeric(d['Flow Pkts/s'],errors='coerce'),
  'byte_count_per_sec':pd.to_numeric(d['Flow Byts/s'],errors='coerce'),'packet_size_avg':pd.to_numeric(d['Pkt Size Avg'],errors='coerce'),
  'flow_duration':pd.to_numeric(d['Flow Duration'],errors='coerce')/1e6,'label':d['target'].map({0:'normal',1:'anomaly'}),
  'is_synthetic':0,'source':'insdn_public_binary','run_id':'insdn_binary_mirror'})
 z=z.dropna(subset=FEATURES+['label']);out.mkdir(parents=True,exist_ok=True);z.to_csv(out/'flow_stats.csv',index=False)
 pd.DataFrame([{'dataset':'insdn_binary','rows':len(z),'normal':int((z.label=='normal').sum()),'anomaly':int((z.label=='anomaly').sum()),'source_file':src.name}]).to_csv(out/'dataset_summary.csv',index=False)

def cic_one(path:Path,notes:list)->pd.DataFrame:
 d=read_frame(path);d.columns=d.columns.str.strip();d['Label']=d['Label'].astype(str).str.strip();d=d[d.Label.isin(LABEL_MAP)].replace([np.inf,-np.inf,'Infinity','inf'],np.nan)
 flags=sum((pd.to_numeric(d[c],errors='coerce').fillna(0) for c in TCP_FLAGS if c in d.columns),start=pd.Series(0,index=d.index,dtype=float))
 has_proto='Protocol' in d; has_src='Source Port' in d
 proto=pd.to_numeric(d['Protocol'],errors='coerce') if has_proto else pd.Series(np.where(flags>0,6,0),index=d.index)
 src_port=pd.to_numeric(d['Source Port'],errors='coerce') if has_src else pd.Series(-1,index=d.index)
 notes.append({'file':path.name,'protocol':'original Protocol column' if has_proto else 'inferred TCP(6)/0 from TCP flags',
               'source_port':'original Source Port column' if has_src else 'sentinel -1 (column absent)'})
 # Always carry the flag-derived surrogate. It is what the MachineLearningCVE
 # release forces, and computing it here lets the counterfactual mirror reuse the
 # very same rows rather than re-deriving a row set that would no longer align.
 return pd.DataFrame({'ip_proto':proto,'tp_src':src_port,'ip_proto_surrogate':pd.Series(np.where(flags>0,6,0),index=d.index),
  'tp_dst':pd.to_numeric(d['Destination Port'],errors='coerce'),
  'packet_count':pd.to_numeric(d['Total Fwd Packets'],errors='coerce').fillna(0)+pd.to_numeric(d['Total Backward Packets'],errors='coerce').fillna(0),
  'byte_count':pd.to_numeric(d['Total Length of Fwd Packets'],errors='coerce').fillna(0)+pd.to_numeric(d['Total Length of Bwd Packets'],errors='coerce').fillna(0),
  'duration_sec':pd.to_numeric(d['Flow Duration'],errors='coerce')/1e6,'packet_count_per_sec':pd.to_numeric(d['Flow Packets/s'],errors='coerce'),
  'byte_count_per_sec':pd.to_numeric(d['Flow Bytes/s'],errors='coerce'),'packet_size_avg':pd.to_numeric(d['Average Packet Size'],errors='coerce'),
  'flow_duration':pd.to_numeric(d['Flow Duration'],errors='coerce')/1e6,'label':d.Label.map(LABEL_MAP),'source_file':path.name})

UNAFFECTED=[c for c in FEATURES if c not in ('ip_proto','tp_src')]

def prepare_cicids(root:Path,out:Path,counterfactual_out:Path|None=None)->None:
 notes=[];frames=[cic_one(locate(root,name),notes) for name in CIC_FILES.values()];z=pd.concat(frames,ignore_index=True).dropna(subset=FEATURES+['label'])
 z=z.drop_duplicates(subset=FEATURES+['label','source_file']);le=LabelEncoder();y=le.fit_transform(z.label);x=z[FEATURES]
 _,xt,_,yt=train_test_split(x,y,test_size=.2,random_state=42,stratify=y);test=pd.DataFrame(xt,columns=FEATURES);test['label']=yt
 out.mkdir(parents=True,exist_ok=True);test.to_csv(out/'test.csv',index=False)
 summary={'dataset':'cicids2017_3class','rows_after_clean':len(z),'rows_test':len(test),'random_state':42,'labels':dict(zip(le.classes_,map(int,le.transform(le.classes_)))),'source_files':[p.name for p in map(lambda n:locate(root,n),CIC_FILES.values())],'field_provenance':notes,
  'protocol_values':sorted(int(v) for v in pd.unique(z.ip_proto)),'source_port_sentinel_rows':int((z.tp_src==-1).sum())}
 (out/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 if counterfactual_out is None: return
 # Row-matched counterfactual: the identical test rows in the identical order,
 # with every other feature and the label untouched, and only the two
 # provenance-sensitive fields overwritten by what MachineLearningCVE forces.
 # Deriving it from `xt` rather than re-running the cleaner is essential, since
 # dropna and drop_duplicates both key on the two columns being changed and
 # would otherwise silently select a different row set.
 cf=pd.DataFrame(xt,columns=FEATURES).copy()
 cf['ip_proto']=z.loc[xt.index,'ip_proto_surrogate'].to_numpy();cf['tp_src']=-1;cf['label']=yt
 left=test[UNAFFECTED+['label']].reset_index(drop=True);right=cf[UNAFFECTED+['label']].reset_index(drop=True)
 if not left.equals(right): raise SystemExit('counterfactual mirror is not row-matched to the original mirror')
 changed=[c for c in FEATURES if not test[c].reset_index(drop=True).equals(cf[c].reset_index(drop=True))]
 counterfactual_out.mkdir(parents=True,exist_ok=True);cf.to_csv(counterfactual_out/'test.csv',index=False)
 (counterfactual_out/'dataset_summary.json').write_text(json.dumps({**summary,
  'dataset':'cicids2017_3class_counterfactual','derived_from':out.name,'rows_test':len(cf),
  'row_matched':True,'columns_changed':changed,
  'protocol_disagreement_rate':float((test.ip_proto.to_numpy()!=cf.ip_proto.to_numpy()).mean()),
  'field_provenance':[{'file':n['file'],'protocol':'flag-derived surrogate substituted for the original column',
                       'source_port':'sentinel -1 substituted for the original column'} for n in notes]},indent=2),encoding='utf8')

def main()->None:
 # Either half can be rebuilt on its own so that a verifier who already holds one
 # checksum-matching prepared file does not need both raw corpora.
 ap=argparse.ArgumentParser();ap.add_argument('--insdn-src',type=Path);ap.add_argument('--cicids-root',type=Path);ap.add_argument('--cicids-out-name',default='cicids2017_mirror');ap.add_argument('--cicids-counterfactual-name',default=None,help='also write a row-matched mirror whose protocol and source port are replaced by the MachineLearningCVE surrogates');ap.add_argument('--out-root',type=Path,default=Path(__file__).resolve().parents[1]/'data');a=ap.parse_args()
 if a.insdn_src is None and a.cicids_root is None: ap.error('supply --insdn-src and/or --cicids-root')
 if a.insdn_src is not None: prepare_insdn(a.insdn_src,a.out_root/'insdn')
 if a.cicids_root is not None: prepare_cicids(a.cicids_root,a.out_root/a.cicids_out_name,a.out_root/a.cicids_counterfactual_name if a.cicids_counterfactual_name else None)
 print(f'Prepared data under {a.out_root}')
if __name__=='__main__': main()
