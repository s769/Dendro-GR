#include "meshReplay.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "checkPoint.h"
#include "dendro.h"
#include "mpi.h"

namespace bssn {
namespace mesh_replay {

namespace {

Mode g_mode = Mode::Off;
std::string g_dir;
MPI_Comm g_comm = MPI_COMM_WORLD;

struct Event {
    DendroIntL step = 0;
    DendroScalar time = 0;
};
std::vector<Event> g_events;
std::size_t g_next = 0;

std::string flags_path(DendroIntL step, int rank, int npes) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/mesh_flags_%lld_%d_%d.bin",
                  g_dir.c_str(), (long long)step, rank, npes);
    return std::string(buf);
}

std::string oct_path(DendroIntL step, int rank, int npes) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/mesh_oct_%lld_%d_%d.oct", g_dir.c_str(),
                  (long long)step, rank, npes);
    return std::string(buf);
}

std::string manifest_path() { return g_dir + "/mesh_history.manifest"; }

int write_u32_vec(const std::string& path, const std::vector<unsigned int>& v) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return 1;
    unsigned int n = (unsigned int)v.size();
    bool ok = std::fwrite(&n, sizeof(n), 1, f) == 1;
    if (ok && n)
        ok = std::fwrite(v.data(), sizeof(unsigned int), n, f) == n;
    ok = (std::fclose(f) == 0) && ok;
    return ok ? 0 : 1;
}

int read_u32_vec(const std::string& path, std::vector<unsigned int>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 1;
    unsigned int n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1) {
        std::fclose(f);
        return 1;
    }
    v.resize(n);
    if (n && std::fread(v.data(), sizeof(unsigned int), n, f) != n) {
        v.clear();
        std::fclose(f);
        return 1;
    }
    std::fclose(f);
    return 0;
}

}  // namespace

Mode mode() { return g_mode; }
const std::string& dir() { return g_dir; }

void init(MPI_Comm comm) {
    g_comm = comm;
    const char* rec = std::getenv("DENDRO_MESH_RECORD");
    const char* rep = std::getenv("DENDRO_MESH_REPLAY");
    if (rec && rep) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (!rank)
            std::cerr << "[mesh_replay] RECORD and REPLAY both set; "
                         "preferring REPLAY\n";
    }
    if (rep) {
        g_mode = Mode::Replay;
        g_dir  = rep;
    } else if (rec) {
        g_mode = Mode::Record;
        g_dir  = rec;
    } else {
        g_mode = Mode::Off;
        g_dir.clear();
        return;
    }

    if (g_mode == Mode::Record) {
        std::string cmd = "mkdir -p \"" + g_dir + "\"";
        std::system(cmd.c_str());
        return;
    }

    g_events.clear();
    g_next = 0;
    std::ifstream in(manifest_path());
    if (!in) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (!rank)
            std::cerr << "[mesh_replay] cannot open " << manifest_path()
                      << std::endl;
        return;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Event e;
        if (!(iss >> e.step >> e.time)) continue;
        g_events.push_back(e);
    }
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (!rank)
        std::cout << "[mesh_replay] loaded " << g_events.size()
                  << " remesh events from " << manifest_path() << std::endl;
}

bool due(DendroIntL step) {
    if (g_mode != Mode::Replay || g_next >= g_events.size()) return false;
    return g_events[g_next].step == step;
}

int record_flags(ot::Mesh* mesh, DendroIntL step, DendroScalar time) {
    if (g_mode != Mode::Record || !mesh) return 0;
    const int rank = mesh->getMPIRankGlobal();
    const int npes = mesh->getMPICommSizeGlobal();
    const std::vector<unsigned int> flags = mesh->getAllRefinementFlags();
    if (write_u32_vec(flags_path(step, rank, npes), flags) != 0) return 1;

    MPI_Barrier(mesh->getMPIGlobalCommunicator());
    if (!rank) {
        std::ofstream out(manifest_path(), std::ios::app);
        if (!out) return 1;
        out.precision(17);
        out << step << " " << time << "\n";
        std::cout << "[mesh_replay] recorded flags step=" << step
                  << " time=" << time << " nFlags=" << flags.size()
                  << std::endl;
    }
    return 0;
}

int apply_due_flags(ot::Mesh* mesh) {
    if (g_mode != Mode::Replay || !mesh || g_next >= g_events.size()) return 1;
    const int rank = mesh->getMPIRankGlobal();
    const int npes = mesh->getMPICommSizeGlobal();
    const DendroIntL step = g_events[g_next].step;
    std::vector<unsigned int> flags;
    if (read_u32_vec(flags_path(step, rank, npes), flags) != 0) {
        if (!rank)
            std::cerr << "[mesh_replay] failed to read flags for step " << step
                      << std::endl;
        return 1;
    }
    if (!mesh->setMeshRefinementFlags(flags)) {
        if (!rank)
            std::cerr << "[mesh_replay] setMeshRefinementFlags failed at step "
                      << step << " (size " << flags.size() << " vs local "
                      << mesh->getNumLocalMeshElements() << ")\n";
        return 1;
    }
    if (!rank)
        std::cout << "[mesh_replay] applied flags step=" << step
                  << " time=" << g_events[g_next].time
                  << " nFlags=" << flags.size() << std::endl;
    ++g_next;
    return 0;
}

int record_oct_snapshot(const ot::Mesh* mesh, DendroIntL step,
                        DendroScalar time) {
    if (g_mode != Mode::Record || !mesh) return 0;
    (void)time;
    const int rank = mesh->getMPIRankGlobal();
    const int npes = mesh->getMPICommSizeGlobal();
    const ot::TreeNode* pNodes =
        mesh->getAllElements().data() + mesh->getElementLocalBegin();
    return io::checkpoint::writeOctToFile(
        oct_path(step, rank, npes).c_str(), pNodes,
        mesh->getNumLocalMeshElements());
}

std::size_t remaining() {
    if (g_mode != Mode::Replay) return 0;
    return (g_next < g_events.size()) ? (g_events.size() - g_next) : 0;
}

}  // namespace mesh_replay
}  // namespace bssn
