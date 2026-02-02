#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#define MPI_PRINT(rank, expr) \
  do { if ((rank) == 0) { std::cout << expr << std::flush; } } while (0)

static constexpr double PI = 3.14159265358979323846;

const double VDW_RADII[] = {
    0.0,
    2.72687, 2.23177, 2.31586, 2.59365, 3.85788, 3.63867, 3.45820, 3.30702,
    3.17852, 3.06419, 2.81853, 2.85443, 4.25094, 4.05819, 3.91835, 3.81252,
    3.72937, 3.65473, 3.60182, 3.21159, 3.11332, 2.99994, 2.97065, 2.85632,
    2.79774, 2.75144, 2.71365, 2.67774, 3.30230, 2.61066, 4.14133, 4.04401,
    3.99677, 3.97315, 3.95803, 3.91268, 3.88717, 3.44025, 3.16057, 2.95175,
    2.99049, 2.88372, 2.83270, 2.79963, 2.76750, 2.73916, 2.97443, 2.69097,
    4.21692, 4.14984, 4.17629, 4.22354, 4.25188, 4.16118, 4.26795, 3.49883,
    3.32781, 3.35993, 3.40718, 3.37789, 3.35143, 3.32592, 3.30041, 3.18230,
    3.26072, 3.23899, 3.22104, 3.20403, 3.18797, 3.17002, 3.43930, 2.96781,
    2.99522, 2.89978, 2.79113, 2.94797, 2.68341, 2.60215, 3.11143, 2.55585,
    4.10732, 4.06008, 4.12905, 4.44936, 4.48810, 4.50227, 4.62983, 3.47426,
    3.28623, 3.20875, 3.23521, 3.20781, 3.23521, 3.23521, 3.19458, 3.14261,
    3.15490, 3.13033, 3.11710, 3.10482, 3.09348, 3.06892, 3.05758, 3.04513,
    3.03268, 3.02023, 3.00778
};

struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    double norm_sq() const { return x * x + y * y + z * z; }
    double norm() const { return std::sqrt(norm_sq()); }
};

static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

struct CubeData {
    int n_atoms = 0;
    int n_grid[3] = {0, 0, 0};
    Vec3 origin;                 // FIX: cube origin
    Vec3 axis_vector[3];         // grid step vectors (Bohr)
    std::vector<int> atom_index; // atomic number
    std::vector<Vec3> atom_pos;  // Bohr
    std::vector<double> V_pot;   // ESP on grid (Hartree/e)
};

struct FilteredGrid {
    std::vector<Vec3> positions;
    std::vector<double> V_pot;
};

static CubeData read_cube(const std::string& filename) {
    CubeData cube;
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << "\n";
        std::exit(1);
    }

    std::string line;
    std::getline(f, line);
    std::getline(f, line);

    // Third line: natoms origin_x origin_y origin_z (Bohr)
    std::getline(f, line);
    {
        std::istringstream iss(line);
        iss >> cube.n_atoms >> cube.origin.x >> cube.origin.y >> cube.origin.z; // FIX
        if (!iss) {
            std::cerr << "Error: Failed to read cube natoms+origin\n";
            std::exit(1);
        }
    }

    // Next 3 lines: n_grid axis_vector
    for (int i = 0; i < 3; i++) {
        std::getline(f, line);
        std::istringstream iss(line);
        iss >> cube.n_grid[i] >> cube.axis_vector[i].x >> cube.axis_vector[i].y >> cube.axis_vector[i].z;
        if (!iss || cube.n_grid[i] <= 0) {
            std::cerr << "Error: Failed to read cube grid axis " << i << "\n";
            std::exit(1);
        }
    }

    cube.atom_index.resize(cube.n_atoms);
    cube.atom_pos.resize(cube.n_atoms);

    // Atom lines: atom_num charge x y z (Bohr)
    for (int i = 0; i < cube.n_atoms; i++) {
        std::getline(f, line);
        std::istringstream iss(line);
        int atom_num = 0;
        double charge = 0.0;
        iss >> atom_num >> charge >> cube.atom_pos[i].x >> cube.atom_pos[i].y >> cube.atom_pos[i].z;
        if (!iss) {
            std::cerr << "Error: Failed to read atom line " << i << "\n";
            std::exit(1);
        }
        cube.atom_index[i] = atom_num;

        // FIX: apply origin to atom positions so atom_pos and grid_pos live in same coordinate system
        cube.atom_pos[i] += cube.origin;
    }

    const int nx = cube.n_grid[0], ny = cube.n_grid[1], nz = cube.n_grid[2];
    const long long n_total = 1LL * nx * ny * nz;
    cube.V_pot.assign((size_t)n_total, 0.0);

    // Read all remaining ESP values
    std::vector<double> file_data;
    file_data.reserve((size_t)n_total);

    double val = 0.0;
    while (f >> val) file_data.push_back(val);

    if ((long long)file_data.size() < n_total) {
        std::cerr << "Error: Not enough volumetric data in cube. Need " << n_total
                  << " got " << file_data.size() << "\n";
        std::exit(1);
    }

    // FIX: cube volumetric data order is z fastest, then y, then x
    // flat index we use everywhere: ix + iy*nx + iz*(nx*ny)
    long long idx = 0;
    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int iz = 0; iz < nz; iz++) {
                const int flat = ix + iy * nx + iz * (nx * ny);
                cube.V_pot[(size_t)flat] = file_data[(size_t)idx++];
            }
        }
    }

    return cube;
}

// Build 27 neighbor shifts for periodic images of atoms (for filtering only)
static std::vector<Vec3> make_neighbor_shifts_27(const Vec3 box_vectors[3]) {
    std::vector<Vec3> shifts;
    shifts.reserve(27);
    for (int kz = -1; kz <= 1; kz++) {
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                shifts.push_back(box_vectors[0] * kx + box_vectors[1] * ky + box_vectors[2] * kz);
            }
        }
    }
    return shifts;
}

static FilteredGrid filter_grid(const CubeData& cube, double vdw_factor, double vdw_max, int rank) {
    const int nx = cube.n_grid[0];
    const int ny = cube.n_grid[1];
    const int nz = cube.n_grid[2];
    const int layer_size = nx * ny;
    const int n_total = nx * ny * nz;

    // Precompute radii
    std::vector<double> raw_radii(cube.n_atoms);
    std::vector<double> vdw_rmin(cube.n_atoms);
    std::vector<double> vdw_rmax(cube.n_atoms);

    double global_max_r = 5.0;
    for (int i = 0; i < cube.n_atoms; i++) {
        int z = cube.atom_index[i];
        raw_radii[i] = (z >= 1 && z <= 118) ? VDW_RADII[z] : 3.0;
        vdw_rmin[i] = vdw_factor * raw_radii[i];
        vdw_rmax[i] = vdw_max * raw_radii[i];
        global_max_r = std::max(global_max_r, vdw_rmax[i]);
    }

    // Box vectors (full cell)
    Vec3 box_vectors[3];
    for (int i = 0; i < 3; i++) box_vectors[i] = cube.axis_vector[i] * cube.n_grid[i];

    // Compute bounding box of the cell corners (in Cartesian) INCLUDING origin
    Vec3 corners[8];
    int cidx = 0;
    for (int ix = 0; ix <= 1; ix++) {
        for (int iy = 0; iy <= 1; iy++) {
            for (int iz = 0; iz <= 1; iz++) {
                corners[cidx++] =
                    cube.origin +
                    box_vectors[0] * ix +
                    box_vectors[1] * iy +
                    box_vectors[2] * iz;
            }
        }
    }

    Vec3 box_min = corners[0], box_max = corners[0];
    for (int i = 1; i < 8; i++) {
        box_min.x = std::min(box_min.x, corners[i].x);
        box_min.y = std::min(box_min.y, corners[i].y);
        box_min.z = std::min(box_min.z, corners[i].z);
        box_max.x = std::max(box_max.x, corners[i].x);
        box_max.y = std::max(box_max.y, corners[i].y);
        box_max.z = std::max(box_max.z, corners[i].z);
    }

    const double safe_margin = global_max_r + 0.1;
    const Vec3 limit_min = box_min - Vec3(safe_margin, safe_margin, safe_margin);
    const Vec3 limit_max = box_max + Vec3(safe_margin, safe_margin, safe_margin);

    // Effective atoms with 27 images (filtering)
    const std::vector<Vec3> neigh_shifts = make_neighbor_shifts_27(box_vectors);

    std::vector<Vec3> eff_pos;
    std::vector<double> eff_rmin;
    std::vector<double> eff_rmax;
    eff_pos.reserve((size_t)cube.n_atoms * neigh_shifts.size());
    eff_rmin.reserve((size_t)cube.n_atoms * neigh_shifts.size());
    eff_rmax.reserve((size_t)cube.n_atoms * neigh_shifts.size());

    for (const auto& sh : neigh_shifts) {
        for (int ai = 0; ai < cube.n_atoms; ai++) {
            Vec3 p = cube.atom_pos[ai] + sh;
            if (p.x >= limit_min.x && p.x <= limit_max.x &&
                p.y >= limit_min.y && p.y <= limit_max.y &&
                p.z >= limit_min.z && p.z <= limit_max.z) {
                eff_pos.push_back(p);
                eff_rmin.push_back(vdw_rmin[ai]);
                eff_rmax.push_back(vdw_rmax[ai]);
            }
        }
    }

    MPI_PRINT(rank, "Grid Points: " << n_total
          << ", Effective Atoms: " << eff_pos.size() << "\n");

    // Precompute base positions for (i,j,0): origin + i*a + j*b
    std::vector<Vec3> base_xy((size_t)layer_size);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            base_xy[(size_t)(i + j * nx)] = cube.origin + cube.axis_vector[0] * i + cube.axis_vector[1] * j;
        }
    }

    std::vector<char> valid_mask((size_t)n_total, 0);
    std::vector<char> near_mask((size_t)n_total, 0);

    // FIX: do NOT assume axis_vector[2] is aligned with Z.
    // Use a conservative slab pre-filter based on distance to the plane perpendicular to c step:
    // We approximate by projecting (atom - mid) onto unit(c_step).
    const Vec3 c_step = cube.axis_vector[2];
    const double c_step_len = std::max(1e-12, c_step.norm());
    const Vec3 c_hat = c_step * (1.0 / c_step_len);
    const double half_thickness = 0.5 * c_step_len;

    const int n_eff = (int)eff_pos.size();

    for (int k = 0; k < nz; k++) {
        const Vec3 layer_origin = cube.origin + c_step * k;
        const Vec3 mid = layer_origin + c_step * 0.5;

        // Determine atoms close enough to this layer (fast reject)
        std::vector<int> local_atoms;
        local_atoms.reserve((size_t)n_eff);
        for (int a = 0; a < n_eff; a++) {
            Vec3 d = eff_pos[a] - mid;
            // signed distance to layer mid-plane along c_hat
            double dist_along_c = std::abs(d.dot(c_hat));
            if (dist_along_c < (half_thickness + global_max_r + 1.0)) {
                local_atoms.push_back(a);
            }
        }

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                const int flat = i + j * nx + k * layer_size;
                const Vec3 gp = base_xy[(size_t)(i + j * nx)] + c_step * k;

                bool valid = true;
                bool near = false;
                for (int idx : local_atoms) {
                    const double dist = (gp - eff_pos[idx]).norm();
                    if (dist <= eff_rmin[idx]) { valid = false; break; }
                    if (dist <= eff_rmax[idx]) near = true;
                }
                valid_mask[(size_t)flat] = valid ? 1 : 0;
                near_mask[(size_t)flat] = near ? 1 : 0;
            }
        }
    }

    // Create filtered arrays and subtract mean ESP over selected points
    FilteredGrid out;
    out.positions.reserve((size_t)n_total / 4);
    out.V_pot.reserve((size_t)n_total / 4);

    double V_sum = 0.0;
    long long cnt = 0;

    for (int flat = 0; flat < n_total; flat++) {
        if (valid_mask[(size_t)flat] && near_mask[(size_t)flat]) {
            int k = flat / layer_size;
            int rem = flat % layer_size;
            int j = rem / nx;
            int i = rem % nx;

            Vec3 pos = cube.origin + cube.axis_vector[0] * i + cube.axis_vector[1] * j + cube.axis_vector[2] * k;
            out.positions.push_back(pos);

            const double v = cube.V_pot[(size_t)flat];
            out.V_pot.push_back(v);

            V_sum += v;
            cnt++;
        }
    }

    if (cnt > 0) {
        const double V_mean = V_sum / (double)cnt;
        for (double& v : out.V_pot) v -= V_mean;
    }

    MPI_PRINT(rank, "  Filtered to " << cnt << " valid grid points\n");
    return out;
}

struct EwaldData {
    double alpha = 0.0;
    double sqrt_alpha = 0.0;
    double R_cutoff = 0.0;
    int fit_flag = 1;

    std::vector<Vec3> kvecs;
    std::vector<double> kcoefs;
    std::vector<Vec3> shifts;

    // self-term for potential basis (per atom)
    double self_term = 0.0;
};

static EwaldData setup_ewald(const Vec3 box_vectors[3], double R_cutoff, int fit_flag) {
    EwaldData e;
    e.fit_flag = fit_flag;
    e.R_cutoff = R_cutoff;

    double volume = box_vectors[0].dot(cross(box_vectors[1], box_vectors[2]));
    volume = std::abs(volume); // FIX: use absolute volume
    if (volume < 1e-12) {
        std::cerr << "Error: Bad cell volume\n";
        std::exit(1);
    }

    e.alpha = (PI / R_cutoff) * (PI / R_cutoff);
    e.sqrt_alpha = std::sqrt(e.alpha);
    e.self_term = 2.0 * e.sqrt_alpha / std::sqrt(PI); // FIX: Ewald self-term magnitude

    if (fit_flag == 0) {
        e.shifts.push_back(Vec3(0, 0, 0));
        return e;
    }

    // Reciprocal lattice vectors
    Vec3 a = box_vectors[0];
    Vec3 b = box_vectors[1];
    Vec3 c = box_vectors[2];
    Vec3 recip0 = cross(b, c) * (2.0 * PI / volume);
    Vec3 recip1 = cross(c, a) * (2.0 * PI / volume);
    Vec3 recip2 = cross(a, b) * (2.0 * PI / volume);

    // Range for real-space shifts (based on cutoff / lattice lengths)
    int nmax[3];
    for (int i = 0; i < 3; i++) {
        double len = box_vectors[i].norm();
        nmax[i] = (int)std::floor(R_cutoff / std::max(1e-12, len)) + 1;
    }

    // k-space vectors
    const int KMAX = 7;
    const int KSQMAX = 49;
    const double beta = 1.0 / (4.0 * e.alpha);

    for (int kx = 0; kx <= KMAX; kx++) {
        for (int ky = -KMAX; ky <= KMAX; ky++) {
            for (int kz = -KMAX; kz <= KMAX; kz++) {
                const int ksq = kx * kx + ky * ky + kz * kz;
                if (ksq == 0 || ksq >= KSQMAX) continue;

                Vec3 kvec = recip0 * kx + recip1 * ky + recip2 * kz;
                double k2 = kvec.dot(kvec);
                if (k2 < 1e-16) continue;

                e.kvecs.push_back(kvec);
                e.kcoefs.push_back((4.0 * PI / volume) * std::exp(-beta * k2) / k2);
            }
        }
    }

    // real-space shifts
    for (int ix = -nmax[0]; ix <= nmax[0]; ix++) {
        for (int iy = -nmax[1]; iy <= nmax[1]; iy++) {
            for (int iz = -nmax[2]; iz <= nmax[2]; iz++) {
                e.shifts.push_back(box_vectors[0] * ix + box_vectors[1] * iy + box_vectors[2] * iz);
            }
        }
    }

    return e;
}

static inline double compute_phi_atom(const Vec3& r_grid, const Vec3& r_atom, const EwaldData& e) {
    Vec3 delta = r_grid - r_atom;

    if (e.fit_flag == 0) {
        double d = delta.norm();
        return (d > 1e-12) ? 1.0 / d : 0.0;
    }

    // reciprocal-space sum
    double phi_recp = 0.0;
    for (size_t k = 0; k < e.kvecs.size(); k++) {
        double kr = delta.dot(e.kvecs[k]);
        phi_recp += std::cos(kr) * e.kcoefs[k];
    }

    // real-space sum
    double phi_real = 0.0;
    const double R2 = e.R_cutoff * e.R_cutoff;

    for (const auto& sh : e.shifts) {
        Vec3 img = r_atom + sh;
        Vec3 d = r_grid - img;
        double r2 = d.norm_sq();
        if (r2 > R2) continue;

        // avoid singularity for self image at r=0
        if (std::abs(sh.x) < 1e-14 && std::abs(sh.y) < 1e-14 && std::abs(sh.z) < 1e-14 && r2 < 1e-24) {
            continue;
        }

        double rr = std::sqrt(r2);
        phi_real += std::erfc(e.sqrt_alpha * rr) / rr;
    }

    // FIX: subtract self-term (constant) from each basis function
    return (phi_real + phi_recp) - e.self_term;
}

// Solve linear system using Gaussian elimination with partial pivoting
static std::vector<double> solve_linear_system(std::vector<std::vector<double>>& A, std::vector<double>& b) {
    const int n = (int)A.size();
    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A[k][i]) > std::abs(A[max_row][i])) max_row = k;
        }
        std::swap(A[i], A[max_row]);
        std::swap(b[i], b[max_row]);

        const double piv = A[i][i];
        if (std::abs(piv) < 1e-14) {
            std::cerr << "Warning: near-singular pivot at row " << i << " (pivot=" << piv << ")\n";
        }

        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / piv;
            b[k] -= factor * b[i];
            for (int j = i; j < n; j++) A[k][j] -= factor * A[i][j];
        }
    }

    std::vector<double> x((size_t)n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) sum -= A[i][j] * x[j];
        x[i] = sum / A[i][i];
    }
    return x;
}

// Simple block partitioning
static inline void partition_range(long long N, int rank, int size, long long& begin, long long& end) {
    long long base = N / size;
    long long rem = N % size;
    begin = rank * base + std::min<long long>(rank, rem);
    end = begin + base + (rank < rem ? 1 : 0);
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parameters (edit or parse args as needed)
    std::string cube_file = "/home/wangc/pyREPEAT/example/water/h2o.cube";
    double vdw_factor = 1.0;
    double vdw_max = 1000.0;
    double R_cutoff = 20.0;
    double q_tot = 0.0;
    int fit_flag = 1; // 1: Ewald periodic, 0: simple Coulomb

    if (argc > 1) cube_file = argv[1];

    if (rank == 0) {
        std::cout << "===================================================================\n";
        std::cout << "REPEAT  Electrostatic potential fitted charges for periodic systems\n";
        std::cout << "            CPU+MPI Fixed Version\n";
        std::cout << "===================================================================\n";
        std::cout << "Reading cube file: " << cube_file << "\n";
    }

    // All ranks read (simple + deterministic); if I/O is heavy, switch to rank0 read + MPI_Bcast
    CubeData cube = read_cube(cube_file);

    // Build cell vectors
    Vec3 box_vectors[3];
    for (int i = 0; i < 3; i++) box_vectors[i] = cube.axis_vector[i] * cube.n_grid[i];

    double volume = std::abs(box_vectors[0].dot(cross(box_vectors[1], box_vectors[2])));

    if (rank == 0) {
        std::cout << "  Atoms: " << cube.n_atoms << "\n";
        std::cout << "  Grid: " << cube.n_grid[0] << " x " << cube.n_grid[1] << " x " << cube.n_grid[2] << "\n";
        std::cout << "  Origin: (" << cube.origin.x << ", " << cube.origin.y << ", " << cube.origin.z << ")\n";
        std::cout << "Real box volume = " << std::fixed << std::setprecision(6) << volume << " bohrs^3\n";
        MPI_PRINT(rank, "Filtering grid points...\n");
    }

    FilteredGrid filtered = filter_grid(cube, vdw_factor, vdw_max, rank);
    const long long n_grid_total = (long long)filtered.positions.size();
    const int n_atoms = cube.n_atoms;

    if (n_grid_total <= 0) {
        if (rank == 0) std::cerr << "Error: no valid grid points after filtering.\n";
        MPI_Finalize();
        return 1;
    }

    // Setup Ewald
    EwaldData ewald = setup_ewald(box_vectors, R_cutoff, fit_flag);

    if (rank == 0) {
        std::cout << "Cutoff radius = " << std::fixed << std::setprecision(6) << R_cutoff << "\n";
        std::cout << "Alpha = " << std::fixed << std::setprecision(6) << ewald.alpha << "\n";
        std::cout << "k-vectors: " << ewald.kvecs.size() << ", shifts: " << ewald.shifts.size() << "\n";
        std::cout << "Total # of valid grid points = " << n_grid_total << "\n";
        MPI_PRINT(rank, "MPI ranks = " << size << "\n");
    }

    // Partition grid among ranks
    long long begin = 0, end = 0;
    partition_range(n_grid_total, rank, size, begin, end);

    // Local accumulators
    std::vector<double> ATA_local((size_t)n_atoms * (size_t)n_atoms, 0.0);
    std::vector<double> ATb_local((size_t)n_atoms, 0.0);
    std::vector<double> sum_phi_local((size_t)n_atoms, 0.0);
    double sum_V_local = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Main loop: compute basis values and accumulate normal equations
    for (long long idx = begin; idx < end; idx++) {
        const Vec3& r = filtered.positions[(size_t)idx];
        const double V = filtered.V_pot[(size_t)idx];

        sum_V_local += V;

        // Compute phi vector for this grid point
        std::vector<double> phi((size_t)n_atoms, 0.0);
        for (int a = 0; a < n_atoms; a++) {
            phi[(size_t)a] = compute_phi_atom(r, cube.atom_pos[(size_t)a], ewald);
        }

        for (int i = 0; i < n_atoms; i++) {
            const double phii = phi[(size_t)i];
            ATb_local[(size_t)i] += phii * V;
            sum_phi_local[(size_t)i] += phii;

            // full matrix (could store symmetric only; keep simple)
            for (int j = 0; j < n_atoms; j++) {
                ATA_local[(size_t)i * (size_t)n_atoms + (size_t)j] += phii * phi[(size_t)j];
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    (void)t1;

    // Reduce to rank 0
    std::vector<double> ATA((size_t)n_atoms * (size_t)n_atoms, 0.0);
    std::vector<double> ATb((size_t)n_atoms, 0.0);
    std::vector<double> sum_phi((size_t)n_atoms, 0.0);
    double sum_V = 0.0;

    MPI_Reduce(ATA_local.data(), ATA.data(), (int)ATA.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(ATb_local.data(), ATb.data(), (int)ATb.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(sum_phi_local.data(), sum_phi.data(), (int)sum_phi.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_V_local, &sum_V, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Mean subtraction in normal equations
        const double invN = 1.0 / (double)n_grid_total;
        std::vector<double> phi_bar((size_t)n_atoms, 0.0);
        for (int i = 0; i < n_atoms; i++) phi_bar[(size_t)i] = sum_phi[(size_t)i] * invN;
        const double V_bar = sum_V * invN;

        std::vector<std::vector<double>> A((size_t)n_atoms, std::vector<double>((size_t)n_atoms, 0.0));
        std::vector<double> b((size_t)n_atoms, 0.0);

        for (int i = 0; i < n_atoms; i++) {
            for (int j = 0; j < n_atoms; j++) {
                A[(size_t)i][(size_t)j] = ATA[(size_t)i * (size_t)n_atoms + (size_t)j]
                                        - (double)n_grid_total * phi_bar[(size_t)i] * phi_bar[(size_t)j];
            }
            b[(size_t)i] = ATb[(size_t)i] - (double)n_grid_total * phi_bar[(size_t)i] * V_bar;
        }

        // Add total charge constraint with Lagrange multiplier
        const int n = n_atoms + 1;
        std::vector<std::vector<double>> A_solv((size_t)n, std::vector<double>((size_t)n, 0.0));
        std::vector<double> b_solv((size_t)n, 0.0);

        for (int i = 0; i < n_atoms; i++) {
            for (int j = 0; j < n_atoms; j++) A_solv[(size_t)i][(size_t)j] = A[(size_t)i][(size_t)j];
            A_solv[(size_t)i][(size_t)n_atoms] = 1.0;
            A_solv[(size_t)n_atoms][(size_t)i] = 1.0;
            b_solv[(size_t)i] = b[(size_t)i];
        }
        b_solv[(size_t)n_atoms] = q_tot;

        std::vector<double> sol = solve_linear_system(A_solv, b_solv);
        std::vector<double> charges(sol.begin(), sol.begin() + n_atoms);

        std::cout << "Fitted charges ordered as within the cube file\n";
        std::cout << "----------------------------------------------\n";
        for (int i = 0; i < n_atoms; i++) {
            int atom_num = cube.atom_index[(size_t)i];
            std::cout << "Charge " << (i + 1) << " of type " << atom_num << " = "
                      << std::fixed << std::setprecision(6) << charges[(size_t)i] << "\n";
        }

        double total_charge = std::accumulate(charges.begin(), charges.end(), 0.0);
        std::cout << "Total charge = " << std::fixed << std::setprecision(6) << total_charge << "\n";
        std::cout << "Normal Termination of REPEAT\n";
    }

    MPI_Finalize();
    return 0;
}

