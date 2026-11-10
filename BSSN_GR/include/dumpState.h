#pragma once
// Debug aid: DENDRO_DUMP_STATE="<dir>:<step>[,<step>...]" writes the local
// evolution variables (raw doubles, var-major, owned nodes only) of each rank
// at the START of the listed steps (= state after that many steps). With one
// rank per run the node order is the mesh order, so CPU and GPU runs on the
// same mesh can be diffed node by node.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include "mesh.h"
#include "dendro.h"
#include "parameters.h"
namespace bssn {
inline bool dump_state_wanted(unsigned int step, std::string& dir) {
    static const char* e = std::getenv("DENDRO_DUMP_STATE");
    if (!e) return false;
    std::string s(e);
    size_t c = s.rfind(':');
    if (c == std::string::npos) return false;
    dir = s.substr(0, c);
    std::stringstream ss(s.substr(c + 1));
    std::string tok;
    while (std::getline(ss, tok, ','))
        if ((unsigned int)std::stoul(tok) == step) return true;
    return false;
}
inline void dump_state(const ot::Mesh* mesh, DendroScalar** evar,
                       unsigned int step, const std::string& dir) {
    if (!mesh->isActive()) return;
    const unsigned int lb = mesh->getNodeLocalBegin(), le = mesh->getNodeLocalEnd();
    char fn[512];
    snprintf(fn, sizeof fn, "%s/state_step%u_rank%d.bin", dir.c_str(), step,
             mesh->getMPIRank());
    FILE* f = fopen(fn, "wb");
    if (!f) return;
    unsigned long n = le - lb, nv = bssn::BSSN_NUM_VARS;
    fwrite(&n, sizeof n, 1, f);
    fwrite(&nv, sizeof nv, 1, f);
    for (unsigned int v = 0; v < bssn::BSSN_NUM_VARS; v++)
        fwrite(evar[v] + lb, sizeof(double), n, f);
    fclose(f);
}
inline void dump_unzip(const ot::Mesh* mesh, const double* unz, unsigned int nv,
                       unsigned int step, const std::string& dir) {
    if (!mesh->isActive()) return;
    char fn[512];
    snprintf(fn, sizeof fn, "%s/unzip_step%u_rank%d.bin", dir.c_str(), step,
             mesh->getMPIRank());
    FILE* f = fopen(fn, "wb");
    if (!f) return;
    const unsigned long usz = mesh->getDegOfFreedomUnZip();
    const std::vector<ot::Block>& bl = mesh->getLocalBlockList();
    unsigned long nb = bl.size(), nvv = nv;
    fwrite(&usz, sizeof usz, 1, f);
    fwrite(&nvv, sizeof nvv, 1, f);
    fwrite(&nb, sizeof nb, 1, f);
    for (const auto& b : bl) {
        unsigned long r[5] = {b.getOffset(), b.getAllocationSzX(),
                              b.getAllocationSzY(), b.getAllocationSzZ(),
                              b.getBlkNodeFlag()};
        fwrite(r, sizeof r, 1, f);
    }
    fwrite(unz, sizeof(double), usz * nv, f);
    fclose(f);
}
// DENDRO_DUMP_RHS="<dir>:<step>[:<k>]": dump the unzipped RHS output of the k-th
// (default 1st = RK stage 1) rhs call at that step. Returns true to dump now.
inline bool dump_rhs_wanted(unsigned long step, std::string& dir, unsigned int* kout = nullptr) {
    static const char* e = std::getenv("DENDRO_DUMP_RHS");
    static long last = -1;
    static unsigned int calls = 0;
    if (!e) return false;
    std::string s(e);
    std::vector<std::string> tok;
    { std::stringstream ss(s); std::string x; while (std::getline(ss, x, ':')) tok.push_back(x); }
    auto isnum = [](const std::string& x) { return !x.empty() && x.find_first_not_of("0123456789") == std::string::npos; };
    unsigned long want_step = 0; unsigned int want_k = 1;
    if (tok.size() >= 3 && isnum(tok[tok.size() - 1]) && isnum(tok[tok.size() - 2])) {
        want_step = std::stoul(tok[tok.size() - 2]); want_k = (unsigned)std::stoul(tok[tok.size() - 1]);
        dir = tok[0]; for (size_t q = 1; q + 2 < tok.size(); q++) dir += ":" + tok[q];
    } else if (tok.size() >= 2 && isnum(tok.back())) {
        want_step = std::stoul(tok.back()); dir = tok[0]; for (size_t q = 1; q + 1 < tok.size(); q++) dir += ":" + tok[q];
    } else return false;
    if (step != want_step) return false;
    if (last != (long)step) { last = (long)step; calls = 0; }
    calls++;
    if (kout) *kout = calls;
    return calls == want_k;
}
}  // namespace bssn
