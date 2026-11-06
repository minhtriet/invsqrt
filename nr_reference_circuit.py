"""Reference Newton-Raphson circuit for Benchmark I (see `verify_submission_invsqrt.py`).

Implements the three-step algorithm of the paper with kickmix arithmetic, for `beta = 1`:

1. Normalize. Count the leading zero *pairs* `s` of `x` into a unary register and shift
   `x` in place to `x_norm = x 2^(2s)`, so that `M = x_norm / 2^(2 n_p)` lies in `[1, 4)`
   and `1 / sqrt(x) = 2^(s - n_p) / sqrt(M)`.
2. Initial guess. `y0 ~ 1 / sqrt(M)` from one of four cubic polynomials, chosen by the top
   three bits of `x_norm` (coefficients loaded by a Clifford network from two AND bits),
   evaluated by Horner's rule on `delta = M - {1, 1.5, 2, 3}` with column-truncated
   multipliers: `r1 = c2 - T(c3 delta)`, `r2 = c1 - T(r1 delta)`, `y0 = c0 - T(r2 delta)`.
3. Newton-Raphson. `p = T(y0 (3 - T(M T(y0^2)))) = 2 / sqrt(M)`, then de-normalize with
   a unary-controlled shift: `y1 = p 2^s`, so `y1 / 2^F = 1 / sqrt(x)` with
   `F = wp + n_p + 1` fractional bits of a turn.

`y1` is then used to phase with one `Z_POW` per bit and steps 1-3 are uncomputed in
reverse. `T(.)` drops every partial-product bit below the output register's least
significant bit, so all arithmetic is exactly reproduced by `fixed_point_model`, which
the tests compare against the simulated circuit and which `main` uses to report the
worst-case relative error over the whole domain.

Usage:
    python invsqrt/nr_reference_circuit.py nr.kmb [--n-p 10] [--format kmb|kmx]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from fractions import Fraction

import kickmix as km
import numpy as np

# Cubic coefficients of the paper for 1/sqrt(x) on [1, 1.5] (`A`, around 1) and
# [1.5, 2] (`B`, around 1.5), with alternating signs supplied by the circuit.
POLY_A = (0.99994132489119882162, 0.49609891915903542303, 0.33261112772430493331, 0.14876762006038398086)
POLY_B = (0.81648515205385221995, 0.27136515484240234115, 0.12756148214815175348, 0.044753028579153842218)


@dataclasses.dataclass(frozen=True)
class Precision:
    """Fractional bit widths of the pipeline's registers.

    The defaults were tuned with `fixed_point_model` for `n_p = 10`: worst case relative
    error `2.84e-8` (`2^-24` is `5.96e-8`) at ~3.9k Toffolis per direction.
    """
    delta_bits: int = 14  # Bits of `delta = M - {1, 1.5, 2, 3}` used by the Horner multipliers (M has 2 n_p).
    coef_bits: int = 17  # Coefficients and Horner intermediates `r1`, `r2`, `y0`.
    square_bits: int = 28  # `y0^2`.
    product_bits: int = 28  # `M y0^2` and `3 - M y0^2` (plus two integer bits).
    result_bits: int = 29  # `p = y0 (3 - M y0^2)` (plus one integer bit).

    def coefficient_table(self) -> list[list[int]]:
        """`table[case][j]` = coefficient magnitude `j` for interval `case`, times `2^coef_bits`.

        Cases are `[1, 1.5)`, `[1.5, 2)`, `[2, 3)`, `[3, 4)`; the last two reuse the
        polynomials on `M / 2` with `delta` left unscaled, which divides coefficient `j` by
        `2^j sqrt(2)`. Coefficients are rounded to nearest.
        """
        cases = [POLY_A, POLY_B,
                 [a / (2**j * math.sqrt(2)) for j, a in enumerate(POLY_A)],
                 [b / (2**j * math.sqrt(2)) for j, b in enumerate(POLY_B)]]
        return [[round(c * 2**self.coef_bits) for c in case] for case in cases]

    def output_bits(self, n_p: int) -> int:
        return self.result_bits + n_p + 1


# Classical model.


def fixed_point_model(n_p: int, prec: Precision, xs: np.ndarray) -> np.ndarray:
    """`y1` as computed by `build_nr_circuit` for each `x` in `xs` (`1 <= x < 2^(2 n_p + 2)`).

    `y1 / 2^prec.output_bits(n_p)` approximates `1 / sqrt(x)`. Raises if an intermediate
    would overflow its register, which the circuit would silently wrap.
    """
    width, d, wc, wsq, wz, wp = 2 * n_p + 2, *dataclasses.astuple(prec)
    xs = np.asarray(xs, dtype=np.int64)
    if not ((1 <= xs) & (xs < 2**width)).all():
        raise ValueError('x must lie in [1, 2^(width - 1)).')

    def bit(v: np.ndarray, i: int) -> np.ndarray:
        return (v >> i) & 1

    def product(rows: list[tuple[np.ndarray, int]], operand: np.ndarray, out_bits: int,
                operand_bits: int, limit: int) -> np.ndarray:
        """`T(sum_rows ctl 2^weight operand)` in units of `2^-out_bits`, checked `< 2^limit`."""
        acc = np.zeros_like(xs)
        for ctl, weight in rows:
            e = out_bits - operand_bits + weight
            acc += ctl * (operand << e if e >= 0 else operand >> -e)
        if (acc >= 2**limit).any():
            raise OverflowError('Intermediate register overflow.')
        return acc

    # 1. Normalize: s leading zero pairs, M = x_norm / 2^(width - 2) in [1, 4).
    s = (width - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    x_norm = xs << (2 * s)
    top, second, third = bit(x_norm, width - 1), bit(x_norm, width - 2), bit(x_norm, width - 3)

    # 2. Initial guess: coefficients by case, delta bits of weight 2^-m for m = 1..d.
    table = np.array(prec.coefficient_table(), dtype=np.int64)
    c0, c1, c2, c3 = (table[np.where(top == 0, third, 2 + second), j] for j in range(4))
    delta = [(top & third, -1)] + [(bit(x_norm, width - 2 - m), -m) for m in range(2, min(d, width - 2) + 1)]
    y0, mult = c0, c3
    for acc in (c2, c1, c0):  # r1, r2, y0.
        y0 = mult = acc - product(delta, mult, wc, wc, wc)
        if ((y0 <= 0) | (y0 >= 2**wc)).any():
            raise OverflowError('Horner intermediate does not fit its register.')

    # 3. Newton-Raphson: y0^2 as diagonal bits plus doubled cross terms.
    y = {k: bit(y0, wc - k) for k in range(1, wc + 1)}  # Bit of weight 2^-k.
    square = np.zeros_like(xs)
    for k in range(1, wc + 1):
        if 2 * k <= wsq:
            square += y[k] << (wsq - 2 * k)
        for l in range(k + 1, wc + 1):
            if k + l - 1 <= wsq:
                square += (y[k] & y[l]) << (wsq - (k + l - 1))
    m_rows = [(bit(x_norm, i), i - width + 2) for i in range(width)]
    z = product(m_rows, square, wz, wsq, wz + 2)
    correction = (3 << wz) - z
    p = product([(y[k], -k) for k in range(1, wc + 1)], correction, wp, wz, wp + 1)
    return p << s


def relative_error(n_p: int, prec: Precision, xs: np.ndarray) -> np.ndarray:
    """`|y1 / 2^F - 1 / sqrt(x)| sqrt(x)` of the model for each `x` in `xs`."""
    exact = 1 / np.sqrt(np.asarray(xs, dtype=np.float64))
    approx = fixed_point_model(n_p, prec, xs) / 2.0**prec.output_bits(n_p)
    return np.abs(approx - exact) / exact


# Circuit construction. Each stage takes `inverse` and emits its own uncomputation:
# additions become subtractions, `init_and` becomes `del_and` (free), and the steps whose
# order matters run backwards.


def _and(builder: km.CircuitBuilder, inverse: bool, control1: km.q | km.array,
         control2: km.q | km.array, target: km.q | km.array) -> None:
    if inverse:
        builder.del_and(control1, control2, target)
    else:
        builder.init_and(control1, control2, target)


def _add_row(builder: km.CircuitBuilder, inverse: bool, control: km.q,
             terms: list[tuple[km.q, int]], target: km.array, scratch: km.array,
             subtract: bool = False) -> None:
    """`target += control * sum(2^index source)` for `(source, index)` in `terms`.

    ANDs `control` with the sources into `scratch` (aligned with `target`), adds the
    scratch slice from the lowest index up (its higher bits are clean), and uncomputes the
    ANDs by measurement.
    """
    sources = km.array([q for q, _ in terms])
    products = km.array([scratch[i] for _, i in terms])
    low = min(i for _, i in terms)
    builder.init_and(control, sources, products)
    if subtract != inverse:
        builder.isub(scratch[low:len(target)], target=target[low:])
    else:
        builder.iadd(scratch[low:len(target)], target=target[low:])
    builder.del_and(control, sources, products)


def _flag_zero_pair(builder: km.CircuitBuilder, inverse: bool, x: km.array, flag: km.q) -> None:
    """`flag := top pair of x == 00`."""
    top_pair = x[-2:]
    builder.x(top_pair)
    _and(builder, inverse, top_pair[0], top_pair[1], flag)
    builder.x(top_pair)


def _normalize(builder: km.CircuitBuilder, inverse: bool, x: km.array, flags: km.array) -> None:
    """`x := x 2^(2s)`, with `flags` the unary count `s` of leading zero pairs.

    Flag k is set iff the top pair is zero after k shifts (then the k lowest pairs are
    already zero), in which case the live even and odd bit lanes each rotate left by one.
    """
    for k in (reversed(range(len(flags))) if inverse else range(len(flags))):
        lanes = (x[2 * k::2], x[2 * k + 1::2])
        if inverse:
            for lane in lanes:
                builder.right_rotate(lane, 1, control=flags[k])
            _flag_zero_pair(builder, inverse, x, flags[k])
        else:
            _flag_zero_pair(builder, inverse, x, flags[k])
            for lane in lanes:
                builder.left_rotate(lane, 1, control=flags[k])


def _select_interval(builder: km.CircuitBuilder, inverse: bool, x: km.array, selectors: km.array) -> None:
    """`selectors := (t_a, t_b, half) = (!top & third, top & second, top & third)`.

    `t_a` and `t_b` select the coefficient set; `half` is the 1/2 bit of `delta`.
    """
    width = len(x)
    top, second, third = x[width - 1], x[width - 2], x[width - 3]
    builder.x(top)
    _and(builder, inverse, top, third, selectors[0])
    builder.x(top)
    _and(builder, inverse, top, km.array([second, third]), selectors[1:])


def _load_coefficients(builder: km.CircuitBuilder, prec: Precision, x: km.array,
                       selectors: km.array, coefs: list[km.array]) -> None:
    """XORs the coefficients of the interval selected by `(top, t_a, t_b)` into `coefs`.

    Clifford, and its own inverse.
    """
    top, t_a, t_b = x[len(x) - 1], selectors[0], selectors[1]
    table = prec.coefficient_table()
    for j, reg in enumerate(coefs):
        for control, value in ((True, table[0][j]), (top, table[0][j] ^ table[2][j]),
                               (t_a, table[0][j] ^ table[1][j]), (t_b, table[2][j] ^ table[3][j])):
            builder.gf2_iadd(value, target=reg, control=control)


def _horner(builder: km.CircuitBuilder, inverse: bool, prec: Precision, x: km.array,
            selectors: km.array, coefs: list[km.array], scratch: km.array) -> None:
    """`c2 -= T(c3 delta)`, `c1 -= T(c2 delta)`, `c0 -= T(c1 delta)`, leaving `y0` in `c0`."""
    width, wc = len(x), len(coefs[0])
    delta = [(selectors[2], 1)]  # (qubit, m) with weight 2^-m: the half bit, then bits of M.
    delta += [(x[width - 2 - m], m) for m in range(2, min(prec.delta_bits, width - 2) + 1)]
    c0, c1, c2, c3 = coefs
    steps = ((c2, c3), (c1, c2), (c0, c1))
    horner_step = builder.mark('horner_step')
    for acc, mult in (reversed(steps) if inverse else steps):
        with horner_step:
            for control, m in delta:
                _add_row(builder, inverse, control, [(mult[i], i - m) for i in range(m, wc)], acc,
                         scratch, subtract=True)


def _square(builder: km.CircuitBuilder, inverse: bool, y0: km.array, square: km.array,
            scratch: km.array) -> None:
    """`square := T(y0^2)`: diagonal bits by CNOT into the clean register, cross terms doubled."""
    wc, wsq = len(y0), len(square)
    y = {k: y0[wc - k] for k in range(1, wc + 1)}  # Bit of weight 2^-k.
    diagonal = [k for k in range(1, wc + 1) if 2 * k <= wsq]
    controls = km.array([y[k] for k in diagonal])
    targets = km.array([square[wsq - 2 * k] for k in diagonal])
    if not inverse:
        builder.cx(controls, targets)
    for k in range(1, wc + 1):
        terms = [(y[l], wsq - (k + l - 1)) for l in range(k + 1, wc + 1) if k + l - 1 <= wsq]
        if terms:
            _add_row(builder, inverse, y[k], terms, square, scratch)
    if inverse:
        builder.cx(controls, targets)


def _multiply_m(builder: km.CircuitBuilder, inverse: bool, x: km.array, square: km.array,
                z: km.array, scratch: km.array) -> None:
    """`z := T(M square)`, where bit i of `x_norm` has weight `2^(i - width + 2)` in `M`."""
    width, wsq, wz = len(x), len(square), len(z) - 2
    for i in range(width):
        e = wz - wsq + i - width + 2
        _add_row(builder, inverse, x[i], [(square[j], j + e) for j in range(max(0, -e), wsq)], z, scratch)


def _correct(builder: km.CircuitBuilder, inverse: bool, z: km.array) -> None:
    """`z := 3 - z`, in two's complement."""
    constant = (3 << (len(z) - 2)) + 1
    if inverse:
        builder.isub(constant, target=z)
        builder.x(z)
    else:
        builder.x(z)
        builder.iadd(constant, target=z)


def _multiply_y0(builder: km.CircuitBuilder, inverse: bool, y0: km.array, z: km.array,
                 p: km.array, scratch: km.array) -> None:
    """`p := T(y0 z)`."""
    wc, wz, wp = len(y0), len(z) - 2, len(p) - 1
    for k in range(1, wc + 1):  # Bit of weight 2^-k.
        e = wp - wz - k
        _add_row(builder, inverse, y0[wc - k], [(z[j], j + e) for j in range(max(0, -e), wz + 2)], p, scratch)


def _denormalize(builder: km.CircuitBuilder, inverse: bool, flags: km.array, y1: km.array) -> None:
    """`y1 := y1 2^s`, one unary-controlled shift per flag."""
    wp = len(y1) - len(flags) - 1
    for k in (reversed(range(len(flags))) if inverse else range(len(flags))):
        lane = y1[k:k + wp + 2]
        if inverse:
            builder.right_rotate(lane, 1, control=flags[k])
        else:
            builder.left_rotate(lane, 1, control=flags[k])


def _compute_y1(builder: km.CircuitBuilder, inverse: bool, prec: Precision, x: km.array,
                flags: km.array, selectors: km.array, coefs: list[km.array], square: km.array,
                z: km.array, y1: km.array, scratch: km.array) -> None:
    """Computes `y1 = 2^F / sqrt(x)` from `x` (`x` ends holding `x_norm`), or uncomputes it."""
    y0, p = coefs[0], y1[:len(y1) - len(flags)]
    with builder.mark('uncompute_y1' if inverse else 'compute_y1'):
        if inverse:
            with builder.mark('denormalize'):
                _denormalize(builder, inverse, flags, y1)
            with builder.mark('multiply_y0'):
                _multiply_y0(builder, inverse, y0, z, p, scratch)
            with builder.mark('correct'):
                _correct(builder, inverse, z)
            with builder.mark('multiply_m'):
                _multiply_m(builder, inverse, x, square, z, scratch)
            with builder.mark('square'):
                _square(builder, inverse, y0, square, scratch)
            with builder.mark('horner'):
                _horner(builder, inverse, prec, x, selectors, coefs, scratch)
            with builder.mark('load_coefficients'):
                _load_coefficients(builder, prec, x, selectors, coefs)
            with builder.mark('select_interval'):
                _select_interval(builder, inverse, x, selectors)
            with builder.mark('normalize'):
                _normalize(builder, inverse, x, flags)
        else:
            with builder.mark('normalize'):
                _normalize(builder, inverse, x, flags)
            with builder.mark('select_interval'):
                _select_interval(builder, inverse, x, selectors)
            with builder.mark('load_coefficients'):
                _load_coefficients(builder, prec, x, selectors, coefs)
            with builder.mark('horner'):
                _horner(builder, inverse, prec, x, selectors, coefs, scratch)
            with builder.mark('square'):
                _square(builder, inverse, y0, square, scratch)
            with builder.mark('multiply_m'):
                _multiply_m(builder, inverse, x, square, z, scratch)
            with builder.mark('correct'):
                _correct(builder, inverse, z)
            with builder.mark('multiply_y0'):
                _multiply_y0(builder, inverse, y0, z, p, scratch)
            with builder.mark('denormalize'):
                _denormalize(builder, inverse, flags, y1)


def _phase_by_bits(builder: km.CircuitBuilder, y: km.array) -> None:
    """Applies `y / 2^len(y)` turns via per-bit `Z_POW`."""
    f = len(y)
    with builder.mark('phase_by_bits'):
        for j in range(f):
            builder.z_pow(y[j], Fraction(2) ** (j - f + 1))  # 2^(j - f) turns.


def build_nr_circuit(n_p: int, *, prec: Precision = Precision(),
                     compute_only: bool = False) -> km.Circuit:
    """The reference circuit; `compute_only` stops after step 3 with `y1` named (for tests)."""
    if n_p < 1:
        raise ValueError(f'n_p must be positive, got {n_p}.')
    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    builder = km.CircuitBuilder()

    def alloc(length: int, name: str) -> km.array:  # Named in compute_only mode, for tests.
        if compute_only:
            return builder.create_quantum_register(length, name=name)
        return builder.alloc_qubits(length)

    x = builder.create_quantum_register(width, name='x')
    y1, flags, selectors = alloc(out_bits, 'y1'), alloc(n_p, 'flags'), alloc(3, 'selectors')
    coefs = [alloc(prec.coef_bits, f'c{j}') for j in range(4)]
    square, z = alloc(prec.square_bits, 'square'), alloc(prec.product_bits + 2, 'z')
    scratch = alloc(max(prec.square_bits, prec.product_bits + 2, prec.result_bits + 1), 'scratch')
    registers = (x, flags, selectors, coefs, square, z, y1, scratch)
    _compute_y1(builder, False, prec, *registers)
    if not compute_only:
        _phase_by_bits(builder, y1)
        _compute_y1(builder, True, prec, *registers)
        for reg in (y1, flags, selectors, *coefs, square, z, scratch):
            builder.free(reg)
    return builder.finish_circuit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', help='Path of the .kmb (or .kmx) file to write.')
    parser.add_argument('--n-p', type=int, default=10, help='Bits per momentum component (default: 10).')
    parser.add_argument('--format', choices=('kmb', 'kmx'), default='kmb',
                        help="Output circuit format: 'kmb' (binary, default) or 'kmx' (text).")
    args = parser.parse_args()

    max_x = 3 * (2**args.n_p - 2)**2
    prec = Precision()
    error = relative_error(args.n_p, prec, np.arange(1, max_x + 1))
    print(f'Model: max relative error 2^{math.log2(error.max()):.2f} at x={error.argmax() + 1} '
          f'(tolerance 2^-24).')
    circuit = build_nr_circuit(args.n_p, prec=prec)
    circuit.to_file(args.output, format=args.format)
    toffolis = circuit.operation_statistics().num_toffoli_like
    print(f'Wrote {args.output} ({args.format}): {toffolis:,} Toffolis, {circuit.num_qubits} qubits.')


if __name__ == '__main__':
    main()
