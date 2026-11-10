#!/usr/bin/env python3
"""Numerically validate a hand-structured Hamiltonian-constraint evaluation against the GENERATED CAHD bracket
(BSSN_GR/src/bssneqs_SSL_HD_dxsq.cpp).  Generated:  chi_rhs_CAHD = -C*DENDRO_18*(chi/12)*B/(dt*(10*DENDRO_18+1))
and CodeGen/bssn.py:  chi_rhs += C*chi*(dxsq/dt)*H,  dxsq = dx^2/(1+10dx^2)   =>  B = -12*H."""
import re, os, sys
import numpy as np
src = open(os.path.join(os.path.dirname(__file__), "../../src/bssneqs_SSL_HD_dxsq.cpp")).read()
defs = {}; order = []
for m in re.finditer(r"const double (DENDRO_\d+)\s*=\s*([^;]*);", src):
    defs[m.group(1)] = m.group(2); order.append(m.group(1))
stmt = re.search(r"chi_rhs\[pp\]\s*=\s*(.*?);\s*\n", src, flags=re.S).group(1)
pre = "-BSSN_CAHD_C*DENDRO_18*DENDRO_640*("
i = stmt.index(pre) + len(pre); depth = 1; j = i
while depth:
    c = stmt[j]; depth += (c == '(') - (c == ')'); j += 1
bracket = stmt[i:j-1]
need = set(re.findall(r"DENDRO_\d+", bracket)); seen = set(); st = list(need)
while st:
    x = st.pop()
    if x in seen: continue
    seen.add(x); st += re.findall(r"DENDRO_\d+", defs[x])
body = [(x, defs[x]) for x in order if x in seen]
def pycode(e):
    e = re.sub(r"\[pp\]", "", e)
    return e.replace("pow(", "np.power(").replace("sqrt(", "np.sqrt(").replace("exp(", "np.exp(")
code = "\n".join(f"{n} = {pycode(e)}" for n, e in body) + f"\nB = {pycode(bracket)}\n"
def eval_generated(env):
    loc = dict(env); loc["np"] = np
    # DENDRO_18 = dx_i^2 appears; caller supplies it
    exec(code, loc)
    return loc["B"]
