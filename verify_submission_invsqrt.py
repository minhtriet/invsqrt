"""Verifier for Benchmark I: phasing by `beta / sqrt(x)` (arXiv:2308.12352).

A submission is a kickmix circuit implementing, for every `x` in `[1, max_x]`,

    |x>|0...0>  ->  exp(2 pi i beta / sqrt(x)) |x>|0...0>

The benchmark instance has `n_p = 10` bits per momentum component: `x = |nu - nu'|^2` with
`max_x = 3 (2^n_p - 2)^2 = 3,133,452`, `width = 2 n_p + 2 = 22`, and `beta = 1`.

Registers:
    x               `width` qubits, little-endian. Must be restored.
`x` must consist of distinct qubits (not classical bits). All other qubits are scratch and
must return to |0>. No other registers are allowed.

Check: each shot is simulated from a basis state `|x>|0>`. `x` must be restored, scratch must
return to |0>, and the tracked phase `theta` (in turns) must be within
`tol(x) = 2^-24 * beta / sqrt(x)` of `beta / sqrt(x)` (mod 1). Phases are compared in 64-bit
fixed point (units of 2^-64 turns): rounding the tracked phase and the target moves the
pass/fail boundary by less than 2^-63 turns, a negligible fraction of `tol(max_x)` (which must
be at least 2^-48 turns).

Every `x` in `[1, max_x]` is run `shots_per_x` times (64 by default, 200M shots for the
benchmark instance), in passes over `x = 1..max_x`. Repeated shots of each `x` probe the seeded
outcomes of measurement-based uncomputation (`HMR` and `R`). `HMR` and `R` outcomes in
`km.Simulator` are seeded with the SHA-256 of the circuit's `.kmb` binary representation
(merged with `--salt` when given). Shots are simulated across CPU cores in batches of
`batch_size`.

Cost (counted across simulated shots in `km.Simulator`):
    Toffolis     Mean Toffoli-like (`CCX`, `CCZ`, `CSWAP`) gates executed per shot.
    Z_POW        Clifford angles are free, T angles cost 1 T, other angles cost
                 `0.56 log2(1/eps) + 5.3` T each (mixed fallback synthesis,
                 arXiv:2203.10064), with `eps` the synthesis error in operator norm up
                 to global phase. That allows a relative phase error of up to `2 eps`
                 radians per rotation. Synthesis error isn't simulated, so the
                 worst case `n_rot * 2 eps` must fit in the tolerance left over by the
                 simulated error, `2 pi min_x (tol(x) - |error(x)|)` radians. The
                 verifier uses the largest `eps` that fits, split evenly.
    C            Toffolis + T / 4.
    Q            `circuit.num_qubits`.
    score        Q * C.

Usage:
    python invsqrt/verify_submission_invsqrt.py submission.kmb [--shots-per-x N] [--salt STR] [--seed N]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import sys
from fractions import Fraction

import numpy as np

import kickmix as km

T_PER_TOFFOLI = 4
ETA = 218  # Electrons in the benchmark system.
N_P = 10  # Bits per momentum component in the benchmark instance.
MASK64 = (1 << 64) - 1  # The check works in units of 2^-64 turns, as uint64 (mod 1 turn).
DEFAULT_BATCH_SIZE = 256
DEFAULT_SHOTS_PER_X = 64
BATCHES_PER_CHUNK = 64


# Problem specification.


@dataclasses.dataclass(frozen=True)
class ProblemSpecInvSqrt:
    """Specification of the Benchmark I problem instance."""
    width: int
    max_x: int
    beta: Fraction
    precision_bits: int
    shots_per_x: int
    system_qubits: int

    @classmethod
    def for_n_p(
        cls,
        n_p: int,
        *,
        beta: Fraction = Fraction(1),
        precision_bits: int = 24,
        shots_per_x: int = DEFAULT_SHOTS_PER_X,
        eta: int = ETA,
    ) -> ProblemSpecInvSqrt:
        """`x = |nu - nu'|^2` for momenta with components in `[-(2^(n_p-1) - 1), 2^(n_p-1) - 1]`,
        so `x <= 3 (2^n_p - 2)^2 < 2^(2 n_p + 2)`."""
        return cls(
            width=2 * n_p + 2,
            max_x=3 * (2**n_p - 2)**2,
            beta=beta,
            precision_bits=precision_bits,
            shots_per_x=shots_per_x,
            system_qubits=3 * eta * n_p,
        )

    @property
    def num_shots(self) -> int:
        return self.shots_per_x * self.max_x



# Mathematical and cost helpers.


def reference_values(spec: ProblemSpecInvSqrt, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`target mod 1` and `tol(x)` for each `x` in `xs`.

    Both are uint64 in units of 2^-64 turns, and come exactly from
    `r = floor(2^65 |beta| / sqrt(x)) = isqrt(floor(2^130 beta^2 / x))`: the target is `r`
    rounded to nearest at bit 0, the tolerance is floored and capped at 1/2 turn. Costs ~1 us
    per `x`, so `verify` tabulates all `x` once.
    """
    p, q = spec.beta.numerator, spec.beta.denominator
    num, den, shift, sign = p * p << 130, q * q, spec.precision_bits + 1, 1 if p >= 0 else -1
    rs = [math.isqrt(num // (den * x)) for x in xs.tolist()]
    target = [sign * ((r + 1) >> 1) & MASK64 for r in rs]
    tol = ([min(r >> shift, 1 << 63) for r in rs] if shift >= 0
           else [min(r << -shift, 1 << 63) for r in rs])
    return np.array(target, dtype=np.uint64), np.array(tol, dtype=np.uint64)


def half_turns_to_units(half_turns: int | Fraction) -> int:
    """The nearest multiple of 2^-64 turns (mod 1 turn) to a phase in half turns."""
    n, d = half_turns.as_integer_ratio()
    return (((n << 64) // d + 1) >> 1) & MASK64


def z_pow_class(half_turns: Fraction) -> str:
    if half_turns % Fraction(1, 2) == 0:
        return 'clifford'
    if half_turns % Fraction(1, 4) == 0:
        return 't'
    return 'rotation'


def rotation_t_count(eps: float) -> float:
    return 0.56 * math.log2(1 / eps) + 5.3


def _circuit_sha256(circuit: km.Circuit) -> str:
    """Returns the SHA-256 hex digest of the canonical `.kmb` serialization of `circuit`."""
    buf = io.BytesIO()
    circuit.to_file(buf, format='kmb')
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _initial_seed(circuit_hash: str, salt: str) -> int:
    """Computes the initial RNG seed from `circuit_hash` and optional `salt`."""
    if not salt:
        return int(circuit_hash, 16)
    return int(hashlib.sha256(f'{circuit_hash}{salt}'.encode('utf-8')).hexdigest(), 16)



def to_bits(values: np.ndarray, width: int) -> np.ndarray:
    return (values[None, :] >> np.arange(width, dtype=np.uint64)[:, None]) & 1 == 1


def from_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint64) << np.arange(len(bits), dtype=np.uint64)[:, None]).sum(axis=0)


# Register validation.


def validate_registers(circuit: km.Circuit, spec: ProblemSpecInvSqrt
                       ) -> tuple[km.array, km.array]:
    """Validates the register schema; returns the `x` and scratch qubits."""
    registers = circuit.register_data
    if len(registers) != circuit.num_registers or list(registers) != ['x']:
        raise ValueError(f'Invalid register schema: {list(registers)}')
    x_reg = list(registers['x'])
    for e in x_reg:
        if not isinstance(e, km.q):
            raise ValueError(f"Register 'x' must contain only qubits, but contains {e!r}.")
    if len(x_reg) != spec.width:
        raise ValueError(f'Expected {spec.width} x qubits, got {len(x_reg)}.')
    io = set(x_reg)
    if len(io) != len(x_reg):
        raise ValueError('The x register must consist of distinct qubits.')
    scratch = km.array([km.q(i) for i in range(circuit.num_qubits) if km.q(i) not in io])
    return km.array(x_reg), scratch


# Shot sampling.


def sample_shots(spec: ProblemSpecInvSqrt, start: int, stop: int) -> np.ndarray:
    """The inputs `x` of shots `start..stop-1`: shot `i` is `x = i mod max_x + 1`."""
    return (np.arange(start, stop, dtype=np.uint64) % np.uint64(spec.max_x)) + np.uint64(1)


# Simulation and verification.


_WORKER_FN = None


def _call_worker(task):
    return _WORKER_FN(task)


def _run_parallel(fn, tasks: list, max_workers: int | None) -> list:
    global _WORKER_FN
    if max_workers is not None and max_workers <= 0:
        raise ValueError(f'max_workers must be positive, got {max_workers}.')
    cpus = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    workers = min(len(tasks), cpus if max_workers is None else max_workers)
    if workers <= 1 or 'fork' not in mp.get_all_start_methods():
        return [fn(t) for t in tasks]
    _WORKER_FN = fn
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
            try:
                return list(pool.map(_call_worker, tasks))
            except Exception:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    finally:
        _WORKER_FN = None


def verify(circuit: km.Circuit, spec: ProblemSpecInvSqrt, *, seed: int | None = None,
           salt: str = '', batch_size: int = DEFAULT_BATCH_SIZE,
           max_workers: int | None = None) -> dict:
    """Verifies `circuit` on `spec.shots_per_x` shots of every `x`; raises on failure.

    `batch_size` shots are simulated together; it doesn't affect which shots are run.
    """
    if spec.max_x <= 0 or spec.shots_per_x <= 0:
        raise ValueError(f'max_x and shots_per_x must be positive, got {spec.max_x} and {spec.shots_per_x}.')
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}.')
    num_shots = spec.num_shots
    if num_shots % batch_size:
        raise ValueError(f'num_shots must be a multiple of batch_size={batch_size}, got {num_shots}.')

    circuit_hash = _circuit_sha256(circuit)
    x_reg, scratch = validate_registers(circuit, spec)

    # Reference values for every x, indexed by x - 1.
    target, tol = reference_values(spec, np.arange(1, spec.max_x + 1, dtype=np.uint64))
    if spec.beta and tol[-1] < 1 << 16:
        raise ValueError('The tolerance is too small for 64-bit phase arithmetic.')
    base_seed = _initial_seed(circuit_hash, salt) if seed is None else int(seed)

    def run_chunk(task: tuple[int, int, int]) -> tuple[int, tuple[float, int], int, int, int]:
        chunk_start, chunk_stop, sim_seed = task
        sim = km.Simulator(
            batch_size=batch_size,
            debug_print=False,
            counting_operations=True,
            seed=sim_seed,
        )
        min_margin = 1 << 64
        worst = (-1.0, 0)
        for start in range(chunk_start, chunk_stop, batch_size):
            xs = sample_shots(spec, start, start + batch_size)
            sim.clear_for_shot()
            sim.write_across_shots(x_reg, to_bits(xs, spec.width))
            sim.do(circuit)

            if not np.array_equal(from_bits(sim.read_across_shots(x_reg)), xs):
                raise AssertionError('x was not restored.')
            if len(scratch):
                dirty = sim.read_across_shots(scratch).any(axis=1)
                if dirty.any():
                    raise AssertionError(
                        f'Scratch qubit {scratch[int(np.argmax(dirty))]} did not return to |0>.')

            # Tracked phase as uint64 in units of 2^-64 turns, so arithmetic is mod 1 turn.
            theta = np.array([half_turns_to_units(sim.read_shot_phase(i)) for i in range(batch_size)],
                             dtype=np.uint64)
            allowed = tol[xs - np.uint64(1)]
            error = (theta - target[xs - np.uint64(1)]).view(np.int64)  # In [-1/2, 1/2) turns.
            used = np.abs(error).view(np.uint64)  # |-2^63| wraps to -2^63, i.e. 2^63 as uint64.
            bad = np.flatnonzero(used > allowed)
            if len(bad):
                i = bad[0]
                raise AssertionError(f'x={xs[i]}: phase error {error[i] / 2**64:.3e} turns '
                                     f'exceeds tolerance {allowed[i] / 2**64:.3e}.')
            ratio = np.divide(used, allowed, out=np.zeros(batch_size), where=allowed > 0)
            min_margin = min(min_margin, int((allowed - used).min()))
            worst = max(worst, (float(ratio.max()), -int(xs[ratio == ratio.max()].min())))

        stats = sim.operation_statistics()
        return (
            min_margin,
            worst,
            int(stats.num_toffoli_like),
            int(stats.num_t_like),
            int(stats.num_z_pow(min_clifford_hierarchy_level=4)),
        )

    chunk_shots = BATCHES_PER_CHUNK * batch_size
    starts = range(0, num_shots, chunk_shots)
    seeds = np.random.SeedSequence(base_seed, pool_size=8).generate_state(4 * len(starts), dtype=np.uint64).reshape(-1, 4)
    tasks = [(s, min(s + chunk_shots, num_shots), int.from_bytes(seeds[i], 'little')) for i, s in enumerate(starts)]
    results = _run_parallel(run_chunk, tasks, max_workers)

    min_margin = min(r[0] for r in results)  # Smallest `tol(x) - |error|` seen, in units of 2^-64 turns.
    worst = max(r[1] for r in results)  # `(|error| / tol(x), -x)` of the worst shot.

    # Cost. Each rotation may add `2 eps` radians; `n_rot * 2 eps <= 2 pi min_margin`.
    toffoli_count = sum(r[2] for r in results) / num_shots
    t_base = sum(r[3] for r in results) / num_shots
    n_rot = sum(r[4] for r in results) / num_shots
    eps = math.pi * min_margin / 2**64 / n_rot if n_rot else None
    if n_rot and not eps > 0:
        raise AssertionError('No tolerance left for rotation synthesis error.')
    t_count = t_base + (n_rot * rotation_t_count(eps) if n_rot else 0)
    toffolis = toffoli_count + t_count / T_PER_TOFFOLI
    return {
        'status': 'PASS',
        'score': circuit.num_qubits * toffolis,
        'toffoli_equivalent': toffolis,
        'toffoli_count': toffoli_count,
        'max_toffoli_count': circuit.operation_statistics().num_toffoli_like,
        't_count': t_base,
        'rotation_count': n_rot,
        'rotation_eps': eps,
        'num_qubits': circuit.num_qubits,
        'reaction_depth': circuit.reaction_depth(),
        'num_ops': len(circuit),
        'verified_shots': num_shots,
        'worst_error_over_tolerance': worst[0],
        'worst_x': -worst[1],
        'circuit_sha256': circuit_hash,
    }


# CLI.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('circuit_file', help='Path to a submitted .kmb or .kmx circuit file.')
    parser.add_argument('--format', choices=('auto', 'kmb', 'kmx'), default='auto',
                        help="Input circuit file format (default: 'auto', auto-detects .kmb or .kmx).")
    parser.add_argument('--n-p', type=int, default=N_P,
                        help=f'Bits per momentum component (default: {N_P}).')
    parser.add_argument('--beta', type=Fraction, default=Fraction(1),
                        help='Phase scaling factor beta (default: 1).')
    parser.add_argument('--precision-bits', type=int, default=24,
                        help='Relative precision bits (default: 24).')
    parser.add_argument('--shots-per-x', type=int, default=DEFAULT_SHOTS_PER_X,
                        help=f'Shots per input value x (default: {DEFAULT_SHOTS_PER_X}).')
    parser.add_argument('--eta', type=int, default=ETA,
                        help=f'Number of electrons for system size (default: {ETA}).')
    parser.add_argument('--width', type=int, default=None,
                        help='Register width in bits (default: 2*n_p + 2).')
    parser.add_argument('--max-x', type=int, default=None,
                        help='Maximum input value (default: 3*(2^n_p - 2)^2).')
    parser.add_argument('--system-qubits', type=int, default=None,
                        help='Accompanying system qubits (default: 3*eta*n_p).')
    parser.add_argument('--salt', type=str, default='',
                        help='Optional salt merged with circuit_sha256 to derive the initial seed (default: empty).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Explicit RNG seed overriding circuit_sha256 and --salt.')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Simulator batch size (default: {DEFAULT_BATCH_SIZE}).')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum worker processes (default: all available CPU cores).')
    args = parser.parse_args()

    n_p = args.n_p
    width = args.width if args.width is not None else 2 * n_p + 2
    max_x = args.max_x if args.max_x is not None else 3 * (2**n_p - 2)**2
    system_qubits = args.system_qubits if args.system_qubits is not None else 3 * args.eta * n_p
    spec = ProblemSpecInvSqrt(
        width=width,
        max_x=max_x,
        beta=args.beta,
        precision_bits=args.precision_bits,
        shots_per_x=args.shots_per_x,
        system_qubits=system_qubits,
    )
    try:
        summary = verify(km.Circuit.from_file(args.circuit_file, format=args.format), spec,
                         seed=args.seed, salt=args.salt, batch_size=args.batch_size,
                         max_workers=args.max_workers)
    except (AssertionError, ValueError, RuntimeError, OSError) as ex:
        print(json.dumps({'status': 'FAIL', 'error': str(ex)}, indent=2))
        sys.exit(1)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
