#!/usr/bin/env python3
"""Experiment harness for Benchmark I (invsqrt). Lives next to nr_reference_circuit.py.

  python run_experiment.py inner   <label>              # full-domain model check only (fast, ~3 s)
  python run_experiment.py build   <label>              # inner check + build circuit + register check + stats
  python run_experiment.py verify  <label> [--shots N]  # official verifier; --shots N (N != 64) = PRELIMINARY
  python run_experiment.py promote <label>              # accept only if official PASS and score < baseline
  python run_experiment.py sweep                        # precision grid: model check + circuit build

It never edits the circuit or the verifier. Outputs go to ./experiments/.
Numbers prefixed est_ are estimates; Q and toffoli come from kickmix; verifier_* come from the verifier.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import kickmix as km
import numpy as np

import verify_submission_invsqrt as vs
from nr_reference_circuit import Precision, build_nr_circuit, relative_error

HERE = Path(__file__).resolve().parent
EXP = HERE / 'experiments'
BASELINE = EXP / 'ACCEPTED.json'  # Not <label>.json, so a run labelled "baseline" cannot collide.
N_P = vs.N_P
SPEC = vs.ProblemSpecInvSqrt.for_n_p(N_P)
MAX_X = SPEC.max_x  # 3,133,452
LIMIT = 2.0**-24


def inner(prec: Precision) -> dict:
    xs = np.arange(1, MAX_X + 1)  # Full domain, never sampled.
    try:
        err = relative_error(N_P, prec, xs)
    except (OverflowError, ValueError) as exc:
        return {'inner_pass': False, 'inner_error': str(exc)}
    worst = float(err.max())
    return {'inner_pass': worst < LIMIT, 'max_rel_err': worst, 'err_over_limit': worst / LIMIT,
            'worst_x': int(xs[err.argmax()])}


def est_cost(q: int, toffoli: float, max_rel_err: float, out_bits: int) -> dict:
    """ESTIMATE of C and Q*C, using the verifier's own formulas but a conservative margin: assumes
    only the Z, S and T-angle bits are non-rotations (n_rot = F - 3) and the worst margin is at
    x = max_x. The verifier's numbers always supersede this."""
    n_rot = out_bits - 3
    margin = (LIMIT - max_rel_err) / math.sqrt(MAX_X)  # In turns.
    eps = math.pi * margin / n_rot  # Same as the verifier: eps = pi * min_margin / n_rot.
    t = 1 + n_rot * vs.rotation_t_count(eps)
    c = toffoli + t / vs.T_PER_TOFFOLI
    return {'est_rotations': n_rot, 'est_eps': eps, 'est_T': t, 'est_C': c, 'est_QxC': q * c}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_metrics(label: str) -> dict:
    p = EXP / f'{label}.json'
    return json.loads(p.read_text()) if p.exists() else {'label': label}


def save(m: dict, stage: str, note: str = '') -> None:
    EXP.mkdir(exist_ok=True)
    (EXP / f'{m["label"]}.json').write_text(json.dumps(m, indent=2, default=str))
    keys = ['inner_pass', 'max_rel_err', 'err_over_limit', 'worst_x', 'register_schema_ok', 'Q', 'toffoli',
            'est_C', 'est_QxC', 'verifier_status', 'verifier_score', 'official', 'sha256']
    print(f'--- {m["label"]} / {stage} ---')
    for k in keys:
        if k in m:
            print(f'{k:>20}: {m[k]}')
    if note:
        print(note)
    with open(EXP / 'log.md', 'a') as f:
        f.write(f'| {time.strftime("%Y-%m-%d %H:%M")} | {m["label"]} | {stage} | '
                + ' | '.join(f'{k}={m[k]}' for k in keys if k in m) + f' | {note} |\n')


def cmd_inner(label: str) -> int:
    m = load_metrics(label)
    m.update(inner(Precision()))
    save(m, 'inner', '' if m['inner_pass'] else 'INNER FAIL: revert, do not build or verify.')
    return 0 if m['inner_pass'] else 1


def cmd_build(label: str) -> int:
    prec = Precision()
    m = load_metrics(label)
    m.update(inner(prec))
    if not m['inner_pass']:
        save(m, 'build', 'INNER FAIL: revert, do not build or verify.')
        return 1
    EXP.mkdir(exist_ok=True)
    kmb = EXP / f'{label}.kmb'
    circuit = build_nr_circuit(N_P, prec=prec)
    circuit.to_file(str(kmb), format='kmb')
    try:
        vs.validate_registers(circuit, SPEC)  # The verifier's own schema check.
        m['register_schema_ok'] = True
    except ValueError as exc:
        m['register_schema_ok'] = False
        save(m, 'build', f'REGISTER SCHEMA FAIL: {exc}')
        return 1
    m.update(Q=circuit.num_qubits, toffoli=circuit.operation_statistics().num_toffoli_like,
             sha256=sha256(kmb), precision=dataclasses.asdict(prec))
    m.update(est_cost(m['Q'], m['toffoli'], m['max_rel_err'], prec.output_bits(N_P)))
    save(m, 'build', 'Scratch cleanliness, input restoration and the true score come from `verify`.')
    return 0


def batch_size_for(num_shots: int) -> int:
    """Largest power of two <= the default batch size that divides num_shots (the verifier requires it)."""
    b = vs.DEFAULT_BATCH_SIZE
    while num_shots % b:
        b //= 2
    return b


def cmd_verify(label: str, shots) -> int:
    kmb = EXP / f'{label}.kmb'
    if not kmb.exists():
        print(f'No {kmb}. Run `build {label}` first.')
        return 1
    shots_per_x = shots or vs.DEFAULT_SHOTS_PER_X
    spec = vs.ProblemSpecInvSqrt.for_n_p(N_P, shots_per_x=shots_per_x)
    official = shots_per_x == vs.DEFAULT_SHOTS_PER_X  # Default seed, salt and batch size as in the CLI.
    try:  # Mirrors main() in verify_submission_invsqrt.py.
        summary = vs.verify(km.Circuit.from_file(str(kmb), format='kmb'), spec,
                            batch_size=batch_size_for(spec.num_shots))
    except (AssertionError, ValueError, RuntimeError, OSError) as ex:
        summary = {'status': 'FAIL', 'error': str(ex)}
    tag = 'official' if official else f'prelim{shots_per_x}'
    (EXP / f'{label}.verify_{tag}.json').write_text(json.dumps(summary, indent=2))
    m = load_metrics(label)
    m.update(official=official, shots_per_x=shots_per_x, verifier_status=summary['status'],
             verifier_score=summary.get('score'), verifier=summary)
    print(json.dumps(summary, indent=2))
    save(m, f'verify_{tag}', '' if official else 'PRELIMINARY: not an official result.')
    return 0 if summary['status'] == 'PASS' else 1


def cmd_promote(label: str) -> int:
    m = load_metrics(label)
    if not (m.get('official') and m.get('verifier_status') == 'PASS' and m.get('inner_pass')):
        print('REFUSED: needs inner pass AND the latest verify run to be the official default run with status PASS.')
        return 1
    score = m['verifier_score']
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        if not score < base['verifier_score']:
            print(f'REJECTED: {score} is not lower than baseline {base["verifier_score"]} ({base["label"]}). Revert.')
            return 1
    BASELINE.write_text(json.dumps(m, indent=2, default=str))
    print(f'ACCEPTED {label} as baseline (score {score}). Now: git commit -am "accept {label} score {score}"')
    return 0


def cmd_sweep() -> int:
    base = Precision()
    fields = ['delta_bits', 'square_bits', 'product_bits', 'result_bits']
    rows = []
    for steps in itertools.product((0, 1), repeat=len(fields)):
        prec = dataclasses.replace(base, **{f: getattr(base, f) - s for f, s in zip(fields, steps)})
        r = inner(prec)
        if not r['inner_pass']:
            print(f'{steps}: inner FAIL ({r.get("inner_error", r.get("err_over_limit"))})')
            continue
        circuit = build_nr_circuit(N_P, prec=prec)
        q, tof = circuit.num_qubits, circuit.operation_statistics().num_toffoli_like
        e = est_cost(q, tof, r['max_rel_err'], prec.output_bits(N_P))
        rows.append((e['est_QxC'], steps, q, tof, r['err_over_limit'], e['est_C']))
    print(f'\nreductions of {fields}; Q and Toffolis from kickmix, C and Q*C ESTIMATED; sorted by est Q*C')
    for qc, steps, q, tof, ratio, c in sorted(rows):
        print(f'{steps}  Q={q}  Toffoli={tof:,.0f}  est_C={c:,.0f}  est_QxC={qc:,.0f}  err={ratio:.0%} of limit')
    print('Next: set the top row in the Precision defaults, then `build <label>` and `verify <label> --shots 4`.')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['inner', 'build', 'verify', 'promote', 'sweep'])
    ap.add_argument('label', nargs='?')
    ap.add_argument('--shots', type=int, default=None, help='Shots per x; anything but 64 is PRELIMINARY.')
    a = ap.parse_args()
    if a.command != 'sweep' and not a.label:
        ap.error('label required')
    return {'inner': lambda: cmd_inner(a.label), 'build': lambda: cmd_build(a.label),
            'verify': lambda: cmd_verify(a.label, a.shots), 'promote': lambda: cmd_promote(a.label),
            'sweep': cmd_sweep}[a.command]()


if __name__ == '__main__':
    sys.exit(main())
