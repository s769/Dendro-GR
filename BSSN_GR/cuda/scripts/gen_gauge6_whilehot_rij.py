#!/usr/bin/env python3
"""Emit gauge6_point_async_whilehot.cpp from CSE base.

While-hot (no restencil):
  - Split chi/gt firsts (long-lived) from seconds (scoped).
  - Open seconds scope at chi load; close right after early RIJ.
  - Move RIJ compute to just after Gt2 load (gt/chi seconds still hot).
  - At section keeps At_rhs only (uses stored DENDRO_RIJ*).

Requires gauge6_point_async_cse.cpp (single christoffel).
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DD = ["chi", "gt0", "gt1", "gt2", "gt3", "gt4", "gt5"]
# Firsts that must outlive the seconds scope (used in Gt_rhs after close).
LONG_FIRSTS = ["K", "Gt0", "Gt1", "Gt2"]


def hoist_first_only_load(text: str, var: str) -> str:
    """Keep load+PT_FUSED but drop local first/ko decls (already hoisted)."""
    pat = re.compile(
        rf"device::__ld_blk_var1_async_issue__<nx>\(su, {var}\);\n"
        rf"device::__ld_blk_var1_async_wait__\(\);\n"
        rf"DEVICE_REAL grad_0_{var}, kograd_0_{var};\n"
        rf"DEVICE_REAL grad_1_{var}, kograd_1_{var};\n"
        rf"DEVICE_REAL grad_2_{var}, kograd_2_{var};\n"
        rf"PT_FUSED_X3<nx>\(su, i, j, k, inv_dx, &grad_0_{var}, nullptr, &kograd_0_{var}\);\n"
        rf"PT_FUSED_Y3<nx>\(su, i, j, k, inv_dy, &grad_1_{var}, nullptr, &kograd_1_{var}\);\n"
        rf"PT_FUSED_Z3<nx>\(su, i, j, k, inv_dz, &grad_2_{var}, nullptr, &kograd_2_{var}\);\n"
        rf"__syncthreads\(\);\n",
        re.MULTILINE,
    )
    repl = (
        f"device::__ld_blk_var1_async_issue__<nx>(su, {var});\n"
        f"device::__ld_blk_var1_async_wait__();\n"
        f"PT_FUSED_X3<nx>(su, i, j, k, inv_dx, &grad_0_{var}, nullptr, &kograd_0_{var});\n"
        f"PT_FUSED_Y3<nx>(su, i, j, k, inv_dy, &grad_1_{var}, nullptr, &kograd_1_{var});\n"
        f"PT_FUSED_Z3<nx>(su, i, j, k, inv_dz, &grad_2_{var}, nullptr, &kograd_2_{var});\n"
        f"__syncthreads();\n"
    )
    new, n = pat.subn(repl, text, count=1)
    if n != 1:
        raise RuntimeError(f"hoist_first_only_load failed for {var} (n={n})")
    return new


def split_firsts_seconds(text: str, var: str) -> str:
    """Rewrite full PT block so firsts/ko are outside-ready; seconds stay
    as non-const decls assigned by PT_FUSED (caller wraps scope)."""
    pat = re.compile(
        rf"device::__ld_blk_var1_async_issue__<nx>\(su, {var}\);\n"
        rf"device::__ld_blk_var1_async_wait__\(\);\n"
        rf"DEVICE_REAL grad_0_{var}, grad2_0_0_{var}, kograd_0_{var};\n"
        rf"DEVICE_REAL grad_1_{var}, grad2_1_1_{var}, kograd_1_{var};\n"
        rf"DEVICE_REAL grad_2_{var}, grad2_2_2_{var}, kograd_2_{var};\n"
        rf"PT_FUSED_X3<nx>\(su, i, j, k, inv_dx, &grad_0_{var}, &grad2_0_0_{var}, &kograd_0_{var}\);\n"
        rf"PT_FUSED_Y3<nx>\(su, i, j, k, inv_dy, &grad_1_{var}, &grad2_1_1_{var}, &kograd_1_{var}\);\n"
        rf"PT_FUSED_Z3<nx>\(su, i, j, k, inv_dz, &grad_2_{var}, &grad2_2_2_{var}, &kograd_2_{var}\);\n"
        rf"const DEVICE_REAL grad2_0_1_{var} = device::__pt_dxy<nx>\(su, i, j, k, inv_dx, inv_dy\);\n"
        rf"const DEVICE_REAL grad2_0_2_{var} = device::__pt_dxz<nx>\(su, i, j, k, inv_dx, inv_dz\);\n"
        rf"const DEVICE_REAL grad2_1_2_{var} = device::__pt_dyz<nx>\(su, i, j, k, inv_dy, inv_dz\);\n"
        rf"__syncthreads\(\);\n",
        re.MULTILINE,
    )
    repl = (
        f"device::__ld_blk_var1_async_issue__<nx>(su, {var});\n"
        f"device::__ld_blk_var1_async_wait__();\n"
        f"DEVICE_REAL grad2_0_0_{var}, grad2_1_1_{var}, grad2_2_2_{var};\n"
        f"DEVICE_REAL grad2_0_1_{var}, grad2_0_2_{var}, grad2_1_2_{var};\n"
        f"PT_FUSED_X3<nx>(su, i, j, k, inv_dx, &grad_0_{var}, &grad2_0_0_{var}, &kograd_0_{var});\n"
        f"PT_FUSED_Y3<nx>(su, i, j, k, inv_dy, &grad_1_{var}, &grad2_1_1_{var}, &kograd_1_{var});\n"
        f"PT_FUSED_Z3<nx>(su, i, j, k, inv_dz, &grad_2_{var}, &grad2_2_2_{var}, &kograd_2_{var});\n"
        f"grad2_0_1_{var} = device::__pt_dxy<nx>(su, i, j, k, inv_dx, inv_dy);\n"
        f"grad2_0_2_{var} = device::__pt_dxz<nx>(su, i, j, k, inv_dx, inv_dz);\n"
        f"grad2_1_2_{var} = device::__pt_dyz<nx>(su, i, j, k, inv_dy, inv_dz);\n"
        f"__syncthreads();\n"
    )
    new, n = pat.subn(repl, text, count=1)
    if n != 1:
        raise RuntimeError(f"split failed for {var} (n={n})")
    return new


def extract_rij_block(text: str) -> tuple[str, str, str]:
    """Return (before, rij_block, after) where rij_block is the Dendro block
    that assigns DENDRO_RIJ0..5 (starts at chi^2 temps, ends at Dendro }}})."""
    # In CSE At section: after christoffel strip, first big block is RIJ preamble
    marker = "        const double DENDRO_0 = pow(chi[pp], 2);"
    i = text.find(marker)
    if i < 0:
        raise RuntimeError("RIJ preamble not found")
    # Walk back to opening "    {\n        // Dendro"
    start = text.rfind("\n    {\n        // Dendro: {{{", 0, i)
    if start < 0:
        raise RuntimeError("RIJ block open not found")
    start += 1  # point at '    {'
    # End after RIJ5's Dendro }}}
    j = text.find("DENDRO_RIJ5 =", i)
    if j < 0:
        raise RuntimeError("RIJ5 not found")
    end = text.find("// Dendro: }}}\n    }", j)
    if end < 0:
        raise RuntimeError("RIJ block close not found")
    end = end + len("// Dendro: }}}\n    }")
    return text[:start], text[start:end], text[end:]


def main() -> None:
    src = (HERE / "gauge6_point_async_cse.cpp").read_text()

    # 1) Hoist chi/gt firsts+ko decls just before chi load; strip from loads.
    chi_load = src.find("device::__ld_blk_var1_async_issue__<nx>(su, chi);")
    if chi_load < 0:
        raise RuntimeError("chi load not found")
    firsts = []
    for var in DD + LONG_FIRSTS:
        firsts.append(
            f"DEVICE_REAL grad_0_{var}, kograd_0_{var};\n"
            f"DEVICE_REAL grad_1_{var}, kograd_1_{var};\n"
            f"DEVICE_REAL grad_2_{var}, kograd_2_{var};\n"
        )
    # Hoist igt/C*/Gtk + RIJ so they survive seconds scope close.
    cse_decl_pat = re.compile(
        r"// CSE: igt/C\*/RIJ/Gtk live across K → Gt → At\n"
        r"((?:    double DENDRO_\w+;\n)+)",
    )
    m = cse_decl_pat.search(src)
    if not m:
        raise RuntimeError("CSE decl block not found")
    cse_decls = m.group(1)
    # Drop RIJ from mid-file CSE decls (re-add up top); keep igt/C*/Gtk text
    cse_decls_no_rij = re.sub(
        r"    double DENDRO_RIJ[0-5];\n", "", cse_decls
    )
    src2 = cse_decl_pat.sub(
        "// CSE decls hoisted before WHILEHOT seconds scope\n", src, count=1
    )
    rij_decl = (
        "double DENDRO_RIJ0, DENDRO_RIJ1, DENDRO_RIJ2;\n"
        "double DENDRO_RIJ3, DENDRO_RIJ4, DENDRO_RIJ5;\n"
    )
    # Re-find chi load after mutation
    chi_load = src2.find("device::__ld_blk_var1_async_issue__<nx>(su, chi);")
    if chi_load < 0:
        raise RuntimeError("chi load not found after CSE decl strip")
    open_scope = (
        "// WHILEHOT: firsts + igt/C*/Gtk/RIJ live past seconds scope\n"
        + "".join(firsts)
        + cse_decls_no_rij
        + rij_decl
        + "{\n"
        + "// seconds scope begin\n"
    )
    src2 = src2[:chi_load] + open_scope + src2[chi_load:]

    # 2) Split each DD load to only declare seconds inside (firsts already hoisted)
    for var in DD:
        src2 = split_firsts_seconds(src2, var)
    # 2b) K/Gt loads: drop local decls (firsts hoisted; used after seconds close)
    for var in LONG_FIRSTS:
        src2 = hoist_first_only_load(src2, var)

    # 3) Extract RIJ block from At; insert after Gt2 syncthreads (before B0)
    before, rij, after = extract_rij_block(src2)
    # Remove rij from At (after)
    src3 = before + "    // WHILEHOT: RIJ computed earlier\n" + after

    gt2_sync = (
        "device::__ld_blk_var1_async_issue__<nx>(su, Gt2);\n"
        "device::__ld_blk_var1_async_wait__();\n"
    )
    # Find Gt2 block end (its __syncthreads)
    g2 = src3.find("device::__ld_blk_var1_async_issue__<nx>(su, Gt2);")
    if g2 < 0:
        raise RuntimeError("Gt2 load not found")
    g2_sync = src3.find("__syncthreads();\n", g2)
    if g2_sync < 0:
        raise RuntimeError("Gt2 sync not found")
    insert_at = g2_sync + len("__syncthreads();\n")
    # Close seconds scope after RIJ
    early = (
        "\n// WHILEHOT: Ricci while gt/chi seconds hot\n"
        + rij
        + "\n} // WHILEHOT: end seconds scope (gt/chi seconds die)\n\n"
    )
    src3 = src3[:insert_at] + early + src3[insert_at:]

    header = (
        "// Auto-generated by gen_gauge6_whilehot_rij.py — early RIJ + scoped "
        "gt/chi seconds (no restencil).\n"
        "// Base: gauge6_point_async_cse.cpp. Bit-id expected vs CSE at ulp.\n"
    )
    if src3.startswith("// Auto-generated"):
        src3 = re.sub(r"^// Auto-generated.*\n(// .*\n)*", header, src3, count=1)
    else:
        src3 = header + src3

    # Sanity
    if src3.count("DENDRO_RIJ0 =") != 1:
        raise RuntimeError(f"RIJ0 assign count={src3.count('DENDRO_RIJ0 =')}")
    if src3.count("WHILEHOT: end seconds scope") != 1:
        raise RuntimeError("seconds scope close missing")
    if src3.count("{") != src3.count("}"):
        raise RuntimeError(
            f"brace imbalance {{={src3.count('{')} }}={src3.count('}')}"
        )

    dst = HERE / "gauge6_point_async_whilehot.cpp"
    dst.write_text(src3)
    print(f"wrote {dst} ({len(src3.splitlines())} lines)")


if __name__ == "__main__":
    main()
