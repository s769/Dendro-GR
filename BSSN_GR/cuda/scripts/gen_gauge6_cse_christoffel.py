#!/usr/bin/env python3
"""Emit gauge6_point_async_cse.cpp — compute igt/C1/C2/C3 once, reuse in Gt+At.

gauge6_point_async.cpp recomputes the identical christoffel body three times
(K, Gt, At scopes). This hoist:
  - declares igt/C*/RIJ/Gtk at fragment scope (live across K→At)
  - keeps the first compute body
  - strips decls + compute from Gt and At scopes (Gtk assign + RIJ stay)

No restencil. Bit-id expected vs unstaged. May trade FLOPs for longer live
ranges — GB200 A/B decides.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_igt_scopes(text: str) -> list[tuple[int, int, int]]:
    """Return [(outer_brace_open, igt_decl_idx, compute_end_idx), ...] for each
    of the three christoffel scopes. Indices are into the full string."""
    decl_pat = re.compile(r"\n\{\n    double DENDRO_igt0;")
    matches = list(decl_pat.finditer(text))
    if len(matches) != 3:
        raise RuntimeError(f"expected 3 igt scopes, found {len(matches)}")
    out = []
    for m in matches:
        open_brace = m.start() + 1  # points at '{'
        # christoffel compute ends at first C3_k2_5 Dendro }}} after decl
        c3 = text.find("DENDRO_C3_k2_5 =", m.end())
        if c3 < 0:
            raise RuntimeError("C3_k2_5 not found after igt decl")
        end_mark = text.find("// Dendro: }}}\n    }", c3)
        if end_mark < 0:
            raise RuntimeError("christoffel end mark not found")
        compute_end = end_mark + len("// Dendro: }}}\n    }")
        out.append((open_brace, m.start() + 1, compute_end))
    return out


def transform(src: str) -> str:
    scopes = _find_igt_scopes(src)
    # Work back-to-front so indices stay valid.
    # At scope (2): remove decls through christoffel compute; leave opening '{'
    # and whatever follows (RIJ / At_rhs).
    # Gt scope (1): same — leave Gtk assign onward.
    # K scope (0): hoist decls before '{'; keep compute inside.

    # --- strip At christoffel (decls + compute), keep outer '{' ---
    at_open, _, at_comp_end = scopes[2]
    # from just after '{' to compute_end inclusive is decls+compute
    # text[at_open] == '{'; skip "{\n"
    at_body_start = at_open + 1
    src = src[:at_body_start] + "\n    // CSE: reuse igt/C* from K scope\n" + src[at_comp_end:]

    # re-find after mutation
    scopes = _find_igt_scopes(src) if False else None  # noqa — recompute manually
    # After At strip, only 2 igt decl blocks remain
    decl_pat = re.compile(r"\n\{\n    double DENDRO_igt0;")
    matches = list(decl_pat.finditer(src))
    if len(matches) != 2:
        raise RuntimeError(f"after At strip expected 2 igt scopes, got {len(matches)}")

    # --- strip Gt christoffel ---
    m = matches[1]
    gt_open = m.start() + 1
    c3 = src.find("DENDRO_C3_k2_5 =", m.end())
    end_mark = src.find("// Dendro: }}}\n    }", c3)
    gt_comp_end = end_mark + len("// Dendro: }}}\n    }")
    gt_body_start = gt_open + 1
    # Gt still needs Gtk assign which follows the christoffel's closing '}'.
    # After compute_end we are past "    }" that closed the C3 inner+outer
    # wait: compute_end includes the '    }' that closes the INNER C3 block only?
    # Structure:
    # {
    #   decls
    #   { igt } { C1 } ... { C3 }  <- compute_end points after last "    }"
    #   { Gtk }
    #   Gt_rhs
    # }
    # Our end_mark is "// Dendro: }}}\n    }" after C3_k2_5 — that's the C3
    # inner block close. Good — Gtk block follows.
    src = src[:gt_body_start] + "\n    // CSE: reuse igt/C* from K scope\n" + src[gt_comp_end:]

    matches = list(decl_pat.finditer(src))
    if len(matches) != 1:
        raise RuntimeError(f"after Gt strip expected 1 igt scope, got {len(matches)}")

    # --- hoist K decls before '{' ---
    m = matches[0]
    k_open = m.start() + 1  # '{'
    # decls run from after "{\n" to just before first "    {" compute block
    # i.e. from "    double DENDRO_igt0;" through "    double DENDRO_Gtk2;\n"
    decl_start = k_open + 2  # skip "{\n"
    first_inner = src.find("\n    {\n        // Dendro: {{{", decl_start)
    if first_inner < 0:
        raise RuntimeError("first christoffel inner block not found")
    decls = src[decl_start:first_inner]
    if "double DENDRO_igt0;" not in decls or "double DENDRO_Gtk2;" not in decls:
        raise RuntimeError("decl block looks wrong")
    # Remove decls from inside brace; insert before brace
    src = (
        src[:k_open]
        + "// CSE: igt/C*/RIJ/Gtk live across K → Gt → At\n"
        + decls
        + "{\n"
        + src[first_inner + 1 :]  # keep leading newline of first_inner via [1:]? 
    )
    # first_inner points at "\n    {"; we want "{\n    {" so:
    # Actually after decls removal we need "{\n" + content starting at first_inner's "\n    {"
    # Let me redo hoist more carefully.
    return src


def transform_v2(src: str) -> str:
    """Strip duplicate christoffel scopes; hoist decls before the K brace."""
    # Match: {\n decls(igt..Gtk) compute(... through C3_k2_5 Dendro block)
    scope_re = re.compile(
        r"\n\{\n"
        r"(    double DENDRO_igt0;\n"
        r"(?:    double DENDRO_\w+;\n)+)"
        r"("
        r"(?:    \{\n        // Dendro: \{\{\{\n.*?// Dendro: \}\}\}\n    \}\n)*?"
        r"    \{\n        // Dendro: \{\{\{\n"
        r".*?DENDRO_C3_k2_5 =.*?"
        r"// Dendro: \}\}\}\n    \}\n"
        r")",
        re.DOTALL,
    )
    matches = list(scope_re.finditer(src))
    if len(matches) != 3:
        raise RuntimeError(
            f"expected 3 christoffel scopes via regex, got {len(matches)}"
        )
    for i, m in enumerate(matches):
        body = m.group(0)
        if "DENDRO_igt0 =" not in body or "DENDRO_C3_k2_5 =" not in body:
            raise RuntimeError(f"scope {i} missing igt/C3 compute")
        if "DENDRO_Gtk0 =" in body or "DENDRO_RIJ0 =" in body:
            raise RuntimeError(f"scope {i} over-matched into Gtk/RIJ")

    for idx in (2, 1):
        m = list(scope_re.finditer(src))[idx]
        replacement = "\n{\n    // CSE: reuse igt/C1/C2/C3 from earlier scope\n"
        src = src[: m.start()] + replacement + src[m.end() :]

    matches = list(scope_re.finditer(src))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected 1 remaining christoffel scope, got {len(matches)}"
        )
    m = matches[0]
    decls = m.group(1)
    compute = m.group(2)
    hoisted = (
        "\n// CSE: igt/C*/RIJ/Gtk live across K → Gt → At\n"
        + decls
        + "{\n"
        + compute
    )
    src = src[: m.start()] + hoisted + src[m.end() :]
    return src


def main() -> None:
    src_path = HERE / "gauge6_point_async.cpp"
    src = src_path.read_text()
    out = transform_v2(src)
    # Sanity: one igt decl, one igt assign, three C3_k2_5 would be bad
    n_decl = out.count("double DENDRO_igt0;")
    n_assign = out.count("DENDRO_igt0 =")
    n_c3 = out.count("DENDRO_C3_k2_5 =")
    if n_decl != 1 or n_assign != 1 or n_c3 != 1:
        raise RuntimeError(
            f"sanity fail: igt_decl={n_decl} igt_assign={n_assign} C3={n_c3}"
        )
    if "DENDRO_Gtk0 =" not in out or "DENDRO_RIJ0 =" not in out:
        raise RuntimeError("Gtk or RIJ assignment missing after CSE")
    header = (
        "// Auto-generated by gen_gauge6_cse_christoffel.py — "
        "igt/C1/C2/C3 computed once, reused in Gt+At.\n"
        "// Bit-identical algebra to gauge6_point_async.cpp; longer C* live ranges.\n"
    )
    # Replace banner lines
    if out.startswith("// Auto-generated"):
        out = re.sub(r"^// Auto-generated.*\n(// .*\n)*", header, out, count=1)
    else:
        out = header + out
    dst = HERE / "gauge6_point_async_cse.cpp"
    dst.write_text(out)
    print(f"wrote {dst} ({len(out.splitlines())} lines; "
          f"igt_decl={n_decl} assign={n_assign} C3={n_c3})")


if __name__ == "__main__":
    main()
