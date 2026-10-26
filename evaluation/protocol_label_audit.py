"""Protocol-code and label associations on the prepared CSVs.

This is analysis of stored integer encodings, not a claim about IANA or
packet-level protocol identity. Counts are computed from the files; they are
not hardcoded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_FIELDS = ('ip_proto', 'tp_src')
UNAFFECTED = [
    'tp_dst', 'packet_count', 'byte_count', 'duration_sec',
    'packet_count_per_sec', 'byte_count_per_sec', 'packet_size_avg',
]
PATHS = {
    'insdn': ROOT / 'data' / 'insdn' / 'flow_stats.csv',
    'tl': ROOT / 'data' / 'cicids2017_mirror' / 'test.csv',
    'cf': ROOT / 'data' / 'cicids2017_mirror_counterfactual' / 'test.csv',
    'mlcve': ROOT / 'data' / 'cicids2017_mirror_mlcve' / 'test.csv',
}


def cic_attack(label) -> pd.Series:
    """CIC test.csv stores LabelEncoder integers: 0=ddos, 1=normal, 2=portscan."""
    return pd.to_numeric(label, errors='coerce').ne(1)


def insdn_attack(label) -> pd.Series:
    return ~label.astype(str).str.lower().str.strip().isin(['normal', 'benign', '0', '0.0'])


def _present() -> dict[str, Path]:
    return {k: p for k, p in PATHS.items() if p.exists()}


def protocol_stats(df: pd.DataFrame, attack: pd.Series) -> dict:
    rows = []
    for code, g in df.groupby(df.ip_proto):
        n = int(len(g))
        n_att = int(attack.loc[g.index].sum())
        rows.append({
            'code': int(code), 'n': n, 'share': n / len(df),
            'n_attack': n_att, 'attack_rate': n_att / n if n else float('nan'),
        })
    return {'n': len(df), 'by_code': rows}


def transition_matrix(tl: pd.DataFrame, cf: pd.DataFrame) -> dict:
    both = pd.DataFrame({'tl': tl.ip_proto.to_numpy(), 'cf': cf.ip_proto.to_numpy(),
                         'attack': cic_attack(tl.label).to_numpy()})
    counts = (both.groupby(['tl', 'cf']).size()
              .reset_index(name='n').sort_values(['tl', 'cf']))
    changed = both[both.tl != both.cf]
    return {
        'cells': [{'tl': int(r.tl), 'cf': int(r.cf), 'n': int(r.n)} for r in counts.itertuples()],
        'n_changed': int(len(changed)),
        'changed_pairs': sorted({(int(a), int(b)) for a, b in zip(changed.tl, changed.cf)}),
        'udp_to_zero': int(((tl.ip_proto == 17) & (cf.ip_proto == 0)).sum()),
        'tl_udp_n': int((tl.ip_proto == 17).sum()),
        'tl_udp_attack_rate': float(cic_attack(tl.loc[tl.ip_proto == 17, 'label']).mean())
            if (tl.ip_proto == 17).any() else float('nan'),
        'disagreement_rate': float((tl.ip_proto != cf.ip_proto).mean()),
    }


def matched_invariants(tl: pd.DataFrame, cf: pd.DataFrame) -> dict:
    differing = sorted(c for c in tl.columns if not tl[c].equals(cf[c]))
    return {
        'same_length': len(tl) == len(cf),
        'label_equal': bool(tl.label.equals(cf.label)),
        'unaffected_equal': all(tl[c].equals(cf[c]) for c in UNAFFECTED if c in tl.columns),
        'differing': differing,
        'cf_protocol_values': sorted(int(v) for v in cf.ip_proto.unique()),
        'cf_source_port_unique': sorted(int(v) for v in cf.tp_src.unique()),
        'cf_sentinel_rows': int((cf.tp_src == -1).sum()),
    }


def tl_attack_protocol_split(tl: pd.DataFrame) -> dict:
    """CIC encoded labels: 0=ddos, 1=normal, 2=portscan. Counts are from the file."""
    lab = pd.to_numeric(tl.label, errors='coerce')
    ddos = lab.eq(0)
    portscan = lab.eq(2)
    code0 = tl.ip_proto.eq(0)
    code0_attack = code0 & lab.ne(1)
    return {
        'n_ddos': int(ddos.sum()),
        'ddos_all_code6': bool(ddos.any() and tl.loc[ddos, 'ip_proto'].eq(6).all()),
        'n_portscan': int(portscan.sum()),
        'n_portscan_code0': int((portscan & code0).sum()),
        'n_portscan_code6': int((portscan & tl.ip_proto.eq(6)).sum()),
        'portscan_codes': sorted(int(v) for v in tl.loc[portscan, 'ip_proto'].unique()),
        'n_code0': int(code0.sum()),
        'n_code0_attack': int(code0_attack.sum()),
        'code0_attacks_are_portscan': bool(code0_attack.any() and lab.loc[code0_attack].eq(2).all()),
        'code0_attack_rate': float(code0_attack.sum() / code0.sum()) if code0.any() else float('nan'),
    }


def insdn_code0(ins: pd.DataFrame) -> dict:
    m = ins.ip_proto == 0
    sub = ins.loc[m]
    return {
        'n': int(m.sum()),
        'share': float(m.mean()),
        'attack_rate': float(insdn_attack(sub.label).mean()) if m.any() else float('nan'),
        'src_port_all_zero': bool(m.any() and (sub.tp_src == 0).all()),
        'dst_port_all_zero': bool(m.any() and (sub.tp_dst == 0).all()),
    }


def audit(root: Path | None = None) -> dict:
    paths = {k: (root / p.relative_to(ROOT) if root is not None else p) for k, p in PATHS.items()}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError('missing prepared CSVs: ' + ', '.join(missing))
    ins, tl, cf, ml = (pd.read_csv(paths[k]) for k in ('insdn', 'tl', 'cf', 'mlcve'))
    return {
        'insdn': protocol_stats(ins, insdn_attack(ins.label)),
        'tl': protocol_stats(tl, cic_attack(tl.label)),
        'cf': protocol_stats(cf, cic_attack(cf.label)),
        'mlcve': protocol_stats(ml, cic_attack(ml.label)),
        'insdn_code0': insdn_code0(ins),
        'tl_attack_codes': tl_attack_protocol_split(tl),
        'transition': transition_matrix(tl, cf),
        'matched': matched_invariants(tl, cf),
    }


def main() -> None:
    print(json.dumps(audit(), indent=2))


if __name__ == '__main__':
    main()
