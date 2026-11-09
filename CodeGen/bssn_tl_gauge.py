#!/usr/bin/env python3
"""
Tier 3.1 #11 — SymPy tangent-linear scaffold for puncture-gauge BSSN.

Differentiates the Dendro-style gauge RHS (α, β⁰, B⁰) treating field values
and their FD symbols (grad_*, kograd_*) as independent, then applies the
product/chain rule in a tangent direction. Verifies against the closed-form
expressions used by the device harnesses (tl_gauge_*, tl_shift_beta,
tl_Bdriver).

Usage (from CodeGen/):
  python3 bssn_tl_gauge.py
  python3 bssn_tl_gauge.py --emit ../BSSN_GR/cuda/include/bssn_tl_gauge.inc.cuh
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sympy import Rational, Symbol, cse, numbered_symbols, simplify, symbols


def _grad(i: int, name: str) -> Symbol:
    return Symbol(f"grad_{i}_{name}")


def _ko(i: int, name: str) -> Symbol:
    return Symbol(f"kograd_{i}_{name}")


def _adv(b, f_name: str):
    """β·∇ f using Dendro advective-derivative symbols."""
    return sum(b[i] * _grad(i, f_name) for i in range(3))


def _ko_sum(name: str, sigma: Symbol):
    return sigma * sum(_ko(i, name) for i in range(3))


def build_gauge_rhs():
    """Production-like gauge6 interior (λ=[1,1,1,1], λf=[1,0], const η)."""
    l0, l1, l2, l3 = symbols("lambda[0] lambda[1] lambda[2] lambda[3]")
    eta, ko_sigma = symbols("eta ko_sigma")
    three_fourths = Rational(3, 4)

    alpha = Symbol("alpha[pp]")
    K = Symbol("K[pp]")
    beta = [Symbol(f"beta{i}[pp]") for i in range(3)]
    B = [Symbol(f"B{i}[pp]") for i in range(3)]
    Gt = [Symbol(f"Gt{i}[pp]") for i in range(3)]
    # Gt_rhs treated as an independent background input for B_rhs
    # (matches Dendro: B_rhs uses the already-computed Gt_rhs).
    Gt_rhs = [Symbol(f"Gt_rhs{i}[pp]") for i in range(3)]

    a_rhs = -2 * alpha * K + l0 * _adv(beta, "alpha") + _ko_sum("alpha", ko_sigma)

    b_rhs0 = (
        three_fourths * B[0]
        + l1 * _adv(beta, "beta0")
        + _ko_sum("beta0", ko_sigma)
    )

    B_rhs0 = (
        -eta * B[0]
        - l3 * _adv(beta, "Gt0")
        + Gt_rhs[0]
        + l2 * _adv(beta, "B0")
        + _ko_sum("B0", ko_sigma)
    )

    chi = Symbol("chi[pp]")
    # χ is the first non-gauge conformal var (weight −2/3 Lie + 2/3 α K χ)
    weight = Rational(-2, 3)
    chi_rhs = (
        _adv(beta, "chi")
        + weight * chi * sum(_grad(i, f"beta{i}") for i in range(3))
        + Rational(2, 3) * chi * alpha * K
        + _ko_sum("chi", ko_sigma)
    )

    # gt00 scaffold: weighted advection + (−2 α At00). Full tensor Lie
    # (gt_ik ∂_j β^k + …) deferred — δα / δAt00 gates are exact; δgt00 is
    # the scalar-weight piece used by tl_gt_test until full Lie emit lands.
    gt0 = Symbol("gt0[pp]")
    At0 = Symbol("At0[pp]")
    gt_rhs0 = (
        _adv(beta, "gt0")
        + weight * gt0 * sum(_grad(i, f"beta{i}") for i in range(3))
        - 2 * alpha * At0
        + _ko_sum("gt0", ko_sigma)
    )

    return {
        "a_rhs": a_rhs,
        "b_rhs0": b_rhs0,
        "B_rhs0": B_rhs0,
        "chi_rhs": chi_rhs,
        "gt_rhs0": gt_rhs0,
        "syms": {
            "l0": l0,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "eta": eta,
            "ko_sigma": ko_sigma,
            "alpha": alpha,
            "K": K,
            "beta": beta,
            "B": B,
            "Gt": Gt,
            "Gt_rhs": Gt_rhs,
            "chi": chi,
            "gt0": gt0,
            "At0": At0,
        },
    }


def tangent_map(base: Symbol, prefix: str = "v_") -> Symbol:
    """Map background symbol name → tangent symbol (same array indexing)."""
    s = str(base)
    if s.endswith("[pp]"):
        stem = s[: -len("[pp]")]
        return Symbol(f"{prefix}{stem}[pp]")
    if s.startswith("grad_"):
        # grad_i_name → grad_i_v_name  (derivative of tangent)
        rest = s[len("grad_") :]
        i, _, name = rest.partition("_")
        return Symbol(f"grad_{i}_{prefix}{name}")
    if s.startswith("kograd_"):
        rest = s[len("kograd_") :]
        i, _, name = rest.partition("_")
        return Symbol(f"kograd_{i}_{prefix}{name}")
    # parameters / Gt_rhs held fixed → tangent 0
    return 0


def directional_derivative(expr, free_syms):
    """J = Σ (∂expr/∂s) * v(s) over independent Dendro symbols."""
    J = 0
    for s in free_syms:
        v = tangent_map(s)
        if v == 0:
            continue
        J += expr.diff(s) * v
    return simplify(J)


def expected_closed_forms(S):
    """Hand TL formulas matching device harnesses (for self-check)."""
    l0, l1, l2, l3 = S["l0"], S["l1"], S["l2"], S["l3"]
    eta, ko_sigma = S["eta"], S["ko_sigma"]
    alpha, K = S["alpha"], S["K"]
    beta = S["beta"]
    chi = S["chi"]

    v_alpha = Symbol("v_alpha[pp]")
    v_beta0 = Symbol("v_beta0[pp]")
    v_B0 = Symbol("v_B0[pp]")
    v_chi = Symbol("v_chi[pp]")
    v_K = Symbol("v_K[pp]")

    # Gate: δ alpha (β, K fixed) — matches tl_gauge_device_test
    Ja_dalpha = (
        -2 * v_alpha * K
        + l0 * _adv(beta, "v_alpha")
        + _ko_sum("v_alpha", ko_sigma)
    )

    # Gate: δ beta0 (B, β1, β2 fixed) — matches tl_shift_beta_test
    # Jb = λ1 (v ∂0 β0 + β·∇ v) + σ KO(v)
    Jb_dbeta0 = (
        l1
        * (
            v_beta0 * _grad(0, "beta0")
            + beta[0] * _grad(0, "v_beta0")
            + beta[1] * _grad(1, "v_beta0")
            + beta[2] * _grad(2, "v_beta0")
        )
        + _ko_sum("v_beta0", ko_sigma)
    )

    # Gate: δ B0 — matches tl_Bdriver_test GATE_dB0
    JB_dB0 = (
        -eta * v_B0
        + l2 * _adv(beta, "v_B0")
        + _ko_sum("v_B0", ko_sigma)
    )

    # Gate: δ beta0 on B_rhs — matches tl_Bdriver_test GATE_dbeta0
    JB_dbeta0 = l2 * v_beta0 * _grad(0, "B0") - l3 * v_beta0 * _grad(0, "Gt0")

    # χ: δ chi (α,β,K fixed) — first non-gauge check target
    weight = Rational(-2, 3)
    Jchi_dchi = (
        _adv(beta, "v_chi")
        + weight * v_chi * sum(_grad(i, f"beta{i}") for i in range(3))
        + Rational(2, 3) * v_chi * alpha * K
        + _ko_sum("v_chi", ko_sigma)
    )

    # χ: δ alpha (chi,β,K fixed)
    Jchi_dalpha = Rational(2, 3) * chi * v_alpha * K

    # χ: δ K
    Jchi_dK = Rational(2, 3) * chi * alpha * v_K

    gt0, At0 = S["gt0"], S["At0"]
    v_gt0 = Symbol("v_gt0[pp]")
    v_At0 = Symbol("v_At0[pp]")
    weight = Rational(-2, 3)
    Jgt_dgt0 = (
        _adv(beta, "v_gt0")
        + weight * v_gt0 * sum(_grad(i, f"beta{i}") for i in range(3))
        + _ko_sum("v_gt0", ko_sigma)
    )
    Jgt_dalpha = -2 * At0 * v_alpha
    Jgt_dAt0 = -2 * alpha * v_At0

    return {
        "a_rhs_dalpha": Ja_dalpha,
        "b_rhs0_dbeta0": Jb_dbeta0,
        "B_rhs0_dB0": JB_dB0,
        "B_rhs0_dbeta0": JB_dbeta0,
        "chi_rhs_dchi": Jchi_dchi,
        "chi_rhs_dalpha": Jchi_dalpha,
        "chi_rhs_dK": Jchi_dK,
        "gt_rhs0_dgt0": Jgt_dgt0,
        "gt_rhs0_dalpha": Jgt_dalpha,
        "gt_rhs0_dAt0": Jgt_dAt0,
    }


def restrict_direction(J, keep_prefix_stems):
    """Zero all tangent symbols whose stem is not in keep_prefix_stems."""
    out = J
    for s in J.free_symbols:
        name = str(s)
        if not (name.startswith("v_") or "v_" in name):
            continue
        # extract stem after v_
        if name.startswith("v_") and name.endswith("[pp]"):
            stem = name[len("v_") : -len("[pp]")]
        elif name.startswith("grad_") and "_v_" in name:
            stem = name.split("_v_", 1)[1]
        elif name.startswith("kograd_") and "_v_" in name:
            stem = name.split("_v_", 1)[1]
        else:
            continue
        if stem not in keep_prefix_stems:
            out = out.xreplace({s: 0})
    return simplify(out)


def emit_cuda_inc(path: Path, tls: dict):
    """Emit a comment+expression include for inspection (not production)."""
    lines = [
        "// Auto-generated by CodeGen/bssn_tl_gauge.py — scaffold only.",
        "// Do not include in production RHS until FD gates pass.",
        "#pragma once",
        "namespace device { namespace tl { namespace gauge_codegen {",
        "",
    ]
    for name, expr in tls.items():
        replacements, reduced = cse([expr], symbols=numbered_symbols("DENDRO_TL_"))
        lines.append(f"// ---- {name} ----")
        for lhs, rhs in replacements:
            lines.append(f"// {lhs} = {rhs};")
        lines.append(f"// {name} = {reduced[0]};")
        lines.append("")
    lines += ["}}}  // namespace device::tl::gauge_codegen", ""]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", type=Path, default=None)
    args = ap.parse_args()

    pack = build_gauge_rhs()
    S = pack["syms"]
    expected = expected_closed_forms(S)

    checks = []

    # Full symbolic TL, then restrict to the harness direction.
    Ja = directional_derivative(pack["a_rhs"], pack["a_rhs"].free_symbols)
    Ja_a = restrict_direction(Ja, {"alpha"})
    checks.append(("a_rhs / δalpha", Ja_a, expected["a_rhs_dalpha"]))

    Jb = directional_derivative(pack["b_rhs0"], pack["b_rhs0"].free_symbols)
    Jb_b0 = restrict_direction(Jb, {"beta0"})
    checks.append(("b_rhs0 / δbeta0", Jb_b0, expected["b_rhs0_dbeta0"]))

    JB = directional_derivative(pack["B_rhs0"], pack["B_rhs0"].free_symbols)
    JB_B0 = restrict_direction(JB, {"B0"})
    checks.append(("B_rhs0 / δB0", JB_B0, expected["B_rhs0_dB0"]))
    JB_b0 = restrict_direction(JB, {"beta0"})
    checks.append(("B_rhs0 / δbeta0", JB_b0, expected["B_rhs0_dbeta0"]))

    Jchi = directional_derivative(pack["chi_rhs"], pack["chi_rhs"].free_symbols)
    checks.append(
        ("chi_rhs / δchi", restrict_direction(Jchi, {"chi"}), expected["chi_rhs_dchi"])
    )
    checks.append(
        (
            "chi_rhs / δalpha",
            restrict_direction(Jchi, {"alpha"}),
            expected["chi_rhs_dalpha"],
        )
    )
    checks.append(
        ("chi_rhs / δK", restrict_direction(Jchi, {"K"}), expected["chi_rhs_dK"])
    )

    Jgt = directional_derivative(pack["gt_rhs0"], pack["gt_rhs0"].free_symbols)
    checks.append(
        ("gt_rhs0 / δgt0", restrict_direction(Jgt, {"gt0"}), expected["gt_rhs0_dgt0"])
    )
    checks.append(
        (
            "gt_rhs0 / δalpha",
            restrict_direction(Jgt, {"alpha"}),
            expected["gt_rhs0_dalpha"],
        )
    )
    checks.append(
        ("gt_rhs0 / δAt0", restrict_direction(Jgt, {"At0"}), expected["gt_rhs0_dAt0"])
    )

    failed = 0
    for tag, got, exp in checks:
        diff = simplify(got - exp)
        ok = diff == 0
        print(f"{'PASS' if ok else 'FAIL'}  {tag}")
        if not ok:
            failed += 1
            print(f"  got: {got}")
            print(f"  exp: {exp}")
            print(f"  Δ:   {diff}")

    if args.emit:
        emit_cuda_inc(
            args.emit,
            {
                "Ja_dalpha": expected["a_rhs_dalpha"],
                "Jb_dbeta0": expected["b_rhs0_dbeta0"],
                "JB_dB0": expected["B_rhs0_dB0"],
                "JB_dbeta0": expected["B_rhs0_dbeta0"],
                "Jchi_dchi": expected["chi_rhs_dchi"],
                "Jchi_dalpha": expected["chi_rhs_dalpha"],
                "Jchi_dK": expected["chi_rhs_dK"],
                "Jgt_dgt0": expected["gt_rhs0_dgt0"],
                "Jgt_dalpha": expected["gt_rhs0_dalpha"],
                "Jgt_dAt0": expected["gt_rhs0_dAt0"],
            },
        )

    print(f"bssn_tl_gauge: {len(checks) - failed}/{len(checks)} checks PASS")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
