
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static constexpr double PI = 3.14159265358979323846;

static inline void die(const std::string& s) {
    std::cerr << "Error: " << s << "\n";
    std::exit(1);
}
static inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream oss;
        oss << what << ": " << cudaGetErrorString(e);
        die(oss.str());
    }
}
static inline void cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << what << ": cublas status " << (int)st;
        die(oss.str());
    }
}

#define MPI_PRINT(rank, expr) do { if ((rank) == 0) { std::cout << expr << std::flush; } } while (0)

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

// ============================= Vec3 ========================================

struct Vec3 {
    double x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    __host__ __device__ Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    __host__ __device__ Vec3 operator*(double s) const { return Vec3(x*s, y*s, z*s); }
    __host__ __device__ Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    __host__ __device__ double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    __host__ __device__ double norm_sq() const { return x*x + y*y + z*z; }
};

static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

// ============================= Cube Data ====================================

struct CubeData {
    int n_atoms = 0;
    int n_grid[3] = {0,0,0};
    Vec3 origin;
    Vec3 axis_vector[3];
    std::vector<int> atom_index;
    std::vector<Vec3> atom_pos;
    std::vector<double> V_pot;
};

struct FilteredGrid {
    std::vector<Vec3> positions;
    std::vector<double> V_pot;
};

static CubeData read_cube(const std::string& filename) {
    CubeData cube;
    std::ifstream f(filename);
    if (!f.is_open()) die("Cannot open file " + filename);

    std::string line;
    std::getline(f, line);
    std::getline(f, line);

    std::getline(f, line);
    {
        std::istringstream iss(line);
        iss >> cube.n_atoms >> cube.origin.x >> cube.origin.y >> cube.origin.z;
        if (!iss) die("Failed to read cube natoms+origin");
    }

    for (int i = 0; i < 3; i++) {
        std::getline(f, line);
        std::istringstream iss(line);
        iss >> cube.n_grid[i] >> cube.axis_vector[i].x >> cube.axis_vector[i].y >> cube.axis_vector[i].z;
        if (!iss || cube.n_grid[i] <= 0) die("Failed to read cube grid axis");
    }

    cube.atom_index.resize((size_t)cube.n_atoms);
    cube.atom_pos.resize((size_t)cube.n_atoms);

    for (int i = 0; i < cube.n_atoms; i++) {
        std::getline(f, line);
        std::istringstream iss(line);
        int atom_num = 0;
        double charge = 0.0;
        iss >> atom_num >> charge >> cube.atom_pos[(size_t)i].x >> cube.atom_pos[(size_t)i].y >> cube.atom_pos[(size_t)i].z;
        if (!iss) die("Failed to read atom line");
        cube.atom_index[(size_t)i] = atom_num;
        cube.atom_pos[(size_t)i] += cube.origin;
    }

    const int nx = cube.n_grid[0], ny = cube.n_grid[1], nz = cube.n_grid[2];
    const long long n_total = 1LL * nx * ny * nz;
    cube.V_pot.assign((size_t)n_total, 0.0);

    std::vector<double> file_data;
    file_data.reserve((size_t)n_total);

    double val = 0.0;
    while (f >> val) file_data.push_back(val);
    if ((long long)file_data.size() < n_total) die("Not enough volumetric data in cube");

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

// ============================= Filter bins ==================================

struct EffAtomSoA {
    std::vector<double> x, y, z;
    std::vector<double> rmin2, rmax2;
};

struct BinGridCSR {
    Vec3 min_corner;
    double bin_size;
    int nx=0, ny=0, nz=0;
    std::vector<int> offsets;
    std::vector<int> indices;
};

static inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

static void build_effective_atoms_and_bins(
    const CubeData& cube,
    double vdw_factor, double vdw_max,
    EffAtomSoA& eff,
    BinGridCSR& bins,
    int rank
) {
    std::vector<double> rmin((size_t)cube.n_atoms), rmax((size_t)cube.n_atoms);
    double global_max_r = 5.0;
    for (int i = 0; i < cube.n_atoms; i++) {
        int z = cube.atom_index[(size_t)i];
        double raw = (z >= 1 && z <= 118) ? VDW_RADII[z] : 3.0;
        rmin[(size_t)i] = vdw_factor * raw;
        rmax[(size_t)i] = vdw_max * raw;
        global_max_r = std::max(global_max_r, rmax[(size_t)i]);
    }

    Vec3 box_vectors[3];
    for (int i = 0; i < 3; i++) box_vectors[i] = cube.axis_vector[i] * cube.n_grid[i];

    Vec3 corners[8];
    int cidx = 0;
    for (int ix = 0; ix <= 1; ix++)
        for (int iy = 0; iy <= 1; iy++)
            for (int iz = 0; iz <= 1; iz++)
                corners[cidx++] = cube.origin + box_vectors[0] * ix + box_vectors[1] * iy + box_vectors[2] * iz;

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

    std::vector<Vec3> neigh;
    neigh.reserve(27);
    for (int kz = -1; kz <= 1; kz++)
        for (int ky = -1; ky <= 1; ky++)
            for (int kx = -1; kx <= 1; kx++)
                neigh.push_back(box_vectors[0] * kx + box_vectors[1] * ky + box_vectors[2] * kz);

    std::vector<double> ex, ey, ez, ermin2, ermax2;
    ex.reserve((size_t)cube.n_atoms * neigh.size());
    ey.reserve((size_t)cube.n_atoms * neigh.size());
    ez.reserve((size_t)cube.n_atoms * neigh.size());
    ermin2.reserve((size_t)cube.n_atoms * neigh.size());
    ermax2.reserve((size_t)cube.n_atoms * neigh.size());

    for (const auto& sh : neigh) {
        for (int ai = 0; ai < cube.n_atoms; ai++) {
            Vec3 p = cube.atom_pos[(size_t)ai] + sh;
            if (p.x >= limit_min.x && p.x <= limit_max.x &&
                p.y >= limit_min.y && p.y <= limit_max.y &&
                p.z >= limit_min.z && p.z <= limit_max.z) {
                ex.push_back(p.x);
                ey.push_back(p.y);
                ez.push_back(p.z);
                ermin2.push_back(rmin[(size_t)ai] * rmin[(size_t)ai]);
                ermax2.push_back(rmax[(size_t)ai] * rmax[(size_t)ai]);
            }
        }
    }

    MPI_PRINT(rank, "Grid Points: " << (cube.n_grid[0]*cube.n_grid[1]*cube.n_grid[2])
                    << ", Effective Atoms: " << ex.size() << "\n");

    bins.min_corner = limit_min;
    bins.bin_size = std::max(1e-6, global_max_r);

    bins.nx = std::max(1, (int)std::ceil((limit_max.x - limit_min.x) / bins.bin_size));
    bins.ny = std::max(1, (int)std::ceil((limit_max.y - limit_min.y) / bins.bin_size));
    bins.nz = std::max(1, (int)std::ceil((limit_max.z - limit_min.z) / bins.bin_size));

    const int nbins = bins.nx * bins.ny * bins.nz;
    bins.offsets.assign((size_t)nbins + 1, 0);

    auto to_bin = [&](double x, double y, double z) {
        int ix = (int)std::floor((x - bins.min_corner.x) / bins.bin_size);
        int iy = (int)std::floor((y - bins.min_corner.y) / bins.bin_size);
        int iz = (int)std::floor((z - bins.min_corner.z) / bins.bin_size);
        ix = clampi(ix, 0, bins.nx - 1);
        iy = clampi(iy, 0, bins.ny - 1);
        iz = clampi(iz, 0, bins.nz - 1);
        return ix + iy * bins.nx + iz * (bins.nx * bins.ny);
    };

    for (int i = 0; i < (int)ex.size(); i++) {
        int b = to_bin(ex[(size_t)i], ey[(size_t)i], ez[(size_t)i]);
        bins.offsets[(size_t)b + 1] += 1;
    }
    for (int b = 0; b < nbins; b++) bins.offsets[(size_t)b + 1] += bins.offsets[(size_t)b];

    bins.indices.assign((size_t)ex.size(), 0);
    std::vector<int> cursor = bins.offsets;

    for (int i = 0; i < (int)ex.size(); i++) {
        int b = to_bin(ex[(size_t)i], ey[(size_t)i], ez[(size_t)i]);
        int pos = cursor[(size_t)b]++;
        bins.indices[(size_t)pos] = i;
    }

    eff.x = std::move(ex);
    eff.y = std::move(ey);
    eff.z = std::move(ez);
    eff.rmin2 = std::move(ermin2);
    eff.rmax2 = std::move(ermax2);
}

__global__ void filter_grid_kernel_bins(
    int nx, int ny, int nz,
    Vec3 origin, Vec3 ax0, Vec3 ax1, Vec3 ax2,
    Vec3 bin_min, double bin_size,
    int bnx, int bny, int bnz,
    const int* __restrict__ bin_offsets,
    const int* __restrict__ bin_indices,
    const double* __restrict__ ax, const double* __restrict__ ay, const double* __restrict__ az,
    const double* __restrict__ rmin2, const double* __restrict__ rmax2,
    unsigned char* __restrict__ keep_mask
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_total = nx * ny * nz;
    if (tid >= n_total) return;

    int k = tid / (nx * ny);
    int rem = tid - k * (nx * ny);
    int j = rem / nx;
    int i = rem - j * nx;

    double gx = origin.x + ax0.x * (double)i + ax1.x * (double)j + ax2.x * (double)k;
    double gy = origin.y + ax0.y * (double)i + ax1.y * (double)j + ax2.y * (double)k;
    double gz = origin.z + ax0.z * (double)i + ax1.z * (double)j + ax2.z * (double)k;

    int bix = (int)floor((gx - bin_min.x) / bin_size);
    int biy = (int)floor((gy - bin_min.y) / bin_size);
    int biz = (int)floor((gz - bin_min.z) / bin_size);
    bix = max(0, min(bnx - 1, bix));
    biy = max(0, min(bny - 1, biy));
    biz = max(0, min(bnz - 1, biz));

    bool valid = true;
    bool near = false;

    for (int dz = -1; dz <= 1 && valid; dz++) {
        int z2 = biz + dz;
        if (z2 < 0 || z2 >= bnz) continue;
        for (int dy = -1; dy <= 1 && valid; dy++) {
            int y2 = biy + dy;
            if (y2 < 0 || y2 >= bny) continue;
            for (int dx = -1; dx <= 1 && valid; dx++) {
                int x2 = bix + dx;
                if (x2 < 0 || x2 >= bnx) continue;

                int b = x2 + y2 * bnx + z2 * (bnx * bny);
                int beg = bin_offsets[b];
                int end = bin_offsets[b + 1];

                for (int t = beg; t < end; t++) {
                    int ai = bin_indices[t];
                    double dxp = gx - ax[ai];
                    double dyp = gy - ay[ai];
                    double dzp = gz - az[ai];
                    double d2 = dxp * dxp + dyp * dyp + dzp * dzp;
                    if (d2 <= rmin2[ai]) { valid = false; break; }
                    if (d2 <= rmax2[ai]) near = true;
                }
            }
        }
    }
    keep_mask[tid] = (valid && near) ? 1 : 0;
}

static FilteredGrid filter_grid_rank0(const CubeData& cube, double vdw_factor, double vdw_max, int rank) {
    const int nx = cube.n_grid[0];
    const int ny = cube.n_grid[1];
    const int nz = cube.n_grid[2];
    const int n_total = nx * ny * nz;

    EffAtomSoA eff;
    BinGridCSR bins;
    build_effective_atoms_and_bins(cube, vdw_factor, vdw_max, eff, bins, rank);

    unsigned char* d_keep = nullptr;
    double *d_ax=nullptr, *d_ay=nullptr, *d_az=nullptr, *d_rmin2=nullptr, *d_rmax2=nullptr;
    int *d_off=nullptr, *d_idx=nullptr;

    cuda_check(cudaMalloc(&d_keep, (size_t)n_total), "cudaMalloc keep");

    cuda_check(cudaMalloc(&d_ax, sizeof(double) * eff.x.size()), "cudaMalloc ax");
    cuda_check(cudaMalloc(&d_ay, sizeof(double) * eff.y.size()), "cudaMalloc ay");
    cuda_check(cudaMalloc(&d_az, sizeof(double) * eff.z.size()), "cudaMalloc az");
    cuda_check(cudaMalloc(&d_rmin2, sizeof(double) * eff.rmin2.size()), "cudaMalloc rmin2");
    cuda_check(cudaMalloc(&d_rmax2, sizeof(double) * eff.rmax2.size()), "cudaMalloc rmax2");

    cuda_check(cudaMemcpy(d_ax, eff.x.data(), sizeof(double) * eff.x.size(), cudaMemcpyHostToDevice), "cpy ax");
    cuda_check(cudaMemcpy(d_ay, eff.y.data(), sizeof(double) * eff.y.size(), cudaMemcpyHostToDevice), "cpy ay");
    cuda_check(cudaMemcpy(d_az, eff.z.data(), sizeof(double) * eff.z.size(), cudaMemcpyHostToDevice), "cpy az");
    cuda_check(cudaMemcpy(d_rmin2, eff.rmin2.data(), sizeof(double) * eff.rmin2.size(), cudaMemcpyHostToDevice), "cpy rmin2");
    cuda_check(cudaMemcpy(d_rmax2, eff.rmax2.data(), sizeof(double) * eff.rmax2.size(), cudaMemcpyHostToDevice), "cpy rmax2");

    cuda_check(cudaMalloc(&d_off, sizeof(int) * bins.offsets.size()), "cudaMalloc offsets");
    cuda_check(cudaMalloc(&d_idx, sizeof(int) * bins.indices.size()), "cudaMalloc indices");
    cuda_check(cudaMemcpy(d_off, bins.offsets.data(), sizeof(int) * bins.offsets.size(), cudaMemcpyHostToDevice), "cpy offsets");
    cuda_check(cudaMemcpy(d_idx, bins.indices.data(), sizeof(int) * bins.indices.size(), cudaMemcpyHostToDevice), "cpy indices");

    int threads = 256;
    int blocks = (n_total + threads - 1) / threads;
    filter_grid_kernel_bins<<<blocks, threads>>>(
        nx, ny, nz,
        cube.origin, cube.axis_vector[0], cube.axis_vector[1], cube.axis_vector[2],
        bins.min_corner, bins.bin_size,
        bins.nx, bins.ny, bins.nz,
        d_off, d_idx,
        d_ax, d_ay, d_az,
        d_rmin2, d_rmax2,
        d_keep
    );
    cuda_check(cudaGetLastError(), "filter_grid_kernel_bins launch");
    cuda_check(cudaDeviceSynchronize(), "filter kernel sync");

    std::vector<unsigned char> keep((size_t)n_total);
    cuda_check(cudaMemcpy(keep.data(), d_keep, (size_t)n_total, cudaMemcpyDeviceToHost), "cpy keep back");

    FilteredGrid out;
    out.positions.reserve((size_t)n_total / 4);
    out.V_pot.reserve((size_t)n_total / 4);

    double V_sum = 0.0;
    long long cnt = 0;

    for (int flat = 0; flat < n_total; flat++) {
        if (!keep[(size_t)flat]) continue;

        int k = flat / (nx * ny);
        int rem = flat - k * (nx * ny);
        int j = rem / nx;
        int i = rem - j * nx;

        Vec3 pos = cube.origin + cube.axis_vector[0] * i + cube.axis_vector[1] * j + cube.axis_vector[2] * k;
        out.positions.push_back(pos);

        double v = cube.V_pot[(size_t)flat];
        out.V_pot.push_back(v);
        V_sum += v;
        cnt++;
    }

    if (cnt > 0) {
        double V_mean = V_sum / (double)cnt;
        for (double& v : out.V_pot) v -= V_mean;
    }

    MPI_PRINT(rank, "  Filtered to " << cnt << " valid grid points\n");

    cudaFree(d_keep);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
    cudaFree(d_rmin2); cudaFree(d_rmax2);
    cudaFree(d_off); cudaFree(d_idx);

    return out;
}

// ============================= Ewald setup + prune ==========================

static inline double aabb_distance_sq(const Vec3& min1, const Vec3& max1, const Vec3& min2, const Vec3& max2) {
    auto axis = [](double a0,double a1,double b0,double b1){
        if (a1 < b0) { double d=b0-a1; return d*d; }
        if (b1 < a0) { double d=a0-b1; return d*d; }
        return 0.0;
    };
    return axis(min1.x,max1.x,min2.x,max2.x)
         + axis(min1.y,max1.y,min2.y,max2.y)
         + axis(min1.z,max1.z,min2.z,max2.z);
}

struct EwaldData {
    double alpha = 0.0;
    double sqrt_alpha = 0.0;
    double R_cutoff = 0.0;
    int fit_flag = 1;
    std::vector<Vec3> kvecs;
    std::vector<double> kcoefs;
    std::vector<Vec3> shifts;
    double self_term = 0.0;
};

static EwaldData setup_ewald_pruned_aabb(
    const Vec3 box_vectors[3],
    double R_cutoff,
    int fit_flag,
    const std::vector<Vec3>& atom_pos,
    int rank
) {
    EwaldData e;
    e.fit_flag = fit_flag;
    e.R_cutoff = R_cutoff;

    double volume = std::abs(box_vectors[0].dot(cross(box_vectors[1], box_vectors[2])));
    if (volume < 1e-12) die("Bad cell volume");

    e.alpha = (PI / R_cutoff) * (PI / R_cutoff);
    e.sqrt_alpha = std::sqrt(e.alpha);
    e.self_term = 2.0 * e.sqrt_alpha / std::sqrt(PI);

    Vec3 a = box_vectors[0], b = box_vectors[1], c = box_vectors[2];
    Vec3 recip0 = cross(b, c) * (2.0 * PI / volume);
    Vec3 recip1 = cross(c, a) * (2.0 * PI / volume);
    Vec3 recip2 = cross(a, b) * (2.0 * PI / volume);

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

    if (fit_flag == 0) {
        e.shifts.push_back(Vec3(0, 0, 0));
        return e;
    }

    if (atom_pos.empty()) die("No atoms");
    Vec3 atom_min = atom_pos[0], atom_max = atom_pos[0];
    for (const auto& p : atom_pos) {
        atom_min.x = std::min(atom_min.x, p.x);
        atom_min.y = std::min(atom_min.y, p.y);
        atom_min.z = std::min(atom_min.z, p.z);
        atom_max.x = std::max(atom_max.x, p.x);
        atom_max.y = std::max(atom_max.y, p.y);
        atom_max.z = std::max(atom_max.z, p.z);
    }

    Vec3 grid_min, grid_max;
    {
        Vec3 corners[8];
        int cidx = 0;
        for (int ix = 0; ix <= 1; ix++)
            for (int iy = 0; iy <= 1; iy++)
                for (int iz = 0; iz <= 1; iz++)
                    corners[cidx++] = box_vectors[0] * ix + box_vectors[1] * iy + box_vectors[2] * iz;

        grid_min = grid_max = corners[0];
        for (int i = 1; i < 8; i++) {
            grid_min.x = std::min(grid_min.x, corners[i].x);
            grid_min.y = std::min(grid_min.y, corners[i].y);
            grid_min.z = std::min(grid_min.z, corners[i].z);
            grid_max.x = std::max(grid_max.x, corners[i].x);
            grid_max.y = std::max(grid_max.y, corners[i].y);
            grid_max.z = std::max(grid_max.z, corners[i].z);
        }
    }

    const double max_shift = R_cutoff + 100.0;
    int nmax[3];
    for (int i = 0; i < 3; i++) {
        double len = std::sqrt(box_vectors[i].norm_sq());
        nmax[i] = (int)std::ceil(max_shift / len);
        nmax[i] = std::max(0, nmax[i]);
    }

    const double R2 = R_cutoff * R_cutoff;
    for (int ix = -nmax[0]; ix <= nmax[0]; ix++) {
        for (int iy = -nmax[1]; iy <= nmax[1]; iy++) {
            for (int iz = -nmax[2]; iz <= nmax[2]; iz++) {
                Vec3 sh = box_vectors[0] * ix + box_vectors[1] * iy + box_vectors[2] * iz;
                Vec3 a_min = atom_min + sh;
                Vec3 a_max = atom_max + sh;
                double d2 = aabb_distance_sq(grid_min, grid_max, a_min, a_max);
                if (d2 <= R2) e.shifts.push_back(sh);
            }
        }
    }

    MPI_PRINT(rank, "Ewald: K=" << e.kvecs.size() << " shifts=" << e.shifts.size() << "\n");
    return e;
}

// ============================= Solve ========================================

static std::vector<double> solve_linear_system(std::vector<std::vector<double>>& A, std::vector<double>& b) {
    const int n = (int)A.size();
    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int k = i + 1; k < n; k++)
            if (std::abs(A[(size_t)k][(size_t)i]) > std::abs(A[(size_t)max_row][(size_t)i])) max_row = k;

        std::swap(A[(size_t)i], A[(size_t)max_row]);
        std::swap(b[(size_t)i], b[(size_t)max_row]);

        double piv = A[(size_t)i][(size_t)i];
        if (std::abs(piv) < 1e-16) std::cerr << "Warning: near-singular pivot at row " << i << "\n";

        for (int k = i + 1; k < n; k++) {
            double factor = A[(size_t)k][(size_t)i] / piv;
            b[(size_t)k] -= factor * b[(size_t)i];
            for (int j = i; j < n; j++) A[(size_t)k][(size_t)j] -= factor * A[(size_t)i][(size_t)j];
        }
    }

    std::vector<double> x((size_t)n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
        double sum = b[(size_t)i];
        for (int j = i + 1; j < n; j++) sum -= A[(size_t)i][(size_t)j] * x[(size_t)j];
        x[(size_t)i] = sum / A[(size_t)i][(size_t)i];
    }
    return x;
}

// ============================= MPI partition =================================

static inline void partition_counts(int n, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign((size_t)size, 0);
    displs.assign((size_t)size, 0);
    int base = n / size;
    int rem = n % size;
    for (int r = 0; r < size; r++) counts[(size_t)r] = base + (r < rem ? 1 : 0);
    for (int r = 1; r < size; r++) displs[(size_t)r] = displs[(size_t)r - 1] + counts[(size_t)r - 1];
}

static int select_gpu(int rank, int requested) {
    int ngpu = 0;
    cuda_check(cudaGetDeviceCount(&ngpu), "cudaGetDeviceCount");
    if (ngpu <= 0) die("No CUDA devices found");
    int gpu = requested;
    if (gpu < 0) gpu = rank % ngpu;
    if (gpu >= ngpu) die("Requested GPU out of range");
    cuda_check(cudaSetDevice(gpu), "cudaSetDevice");
    return gpu;
}

// ============================= Args =========================================

struct Args {
    std::string cube_file;
    int gpu_id = -1;
    double vdw_factor = 1.0;
    double vdw_max = 1000.0;
    double R_cutoff = 20.0;
    double q_tot = 0.0;
    int fit_flag = 1;
    int chunk_size = 50000;
    int fp64 = 0;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) die("Usage: ./repeat file.cube [options]");
    a.cube_file = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string s = argv[i];
        auto need = [&](const char* opt){ if (i+1 >= argc) die(std::string("Missing value for ")+opt); };
        if (s == "--gpu") { need("--gpu"); a.gpu_id = std::stoi(argv[++i]); }
        else if (s == "--chunk") { need("--chunk"); a.chunk_size = std::stoi(argv[++i]); }
        else if (s == "--fit") { need("--fit"); a.fit_flag = std::stoi(argv[++i]); }
        else if (s == "--qtot") { need("--qtot"); a.q_tot = std::stod(argv[++i]); }
        else if (s == "--vdw_factor") { need("--vdw_factor"); a.vdw_factor = std::stod(argv[++i]); }
        else if (s == "--vdw_max") { need("--vdw_max"); a.vdw_max = std::stod(argv[++i]); }
        else if (s == "--rcut") { need("--rcut"); a.R_cutoff = std::stod(argv[++i]); }
        else if (s == "--fp64") { a.fp64 = 1; }
        else die("Unknown option: " + s);
    }
    return a;
}

// ============================= Trig cache kernels (float) ====================

__global__ void compute_grid_trig_kernel_f32(
    const double* __restrict__ grid_xyz, // [M*3]
    int M,
    const double* __restrict__ kvecs,    // [K*3]
    int K,
    int fp64,
    float* __restrict__ grid_c,          // [K*M]
    float* __restrict__ grid_s           // [K*M]
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || k >= K) return;

    double gx = grid_xyz[3 * (size_t)m + 0];
    double gy = grid_xyz[3 * (size_t)m + 1];
    double gz = grid_xyz[3 * (size_t)m + 2];

    double kx = kvecs[3 * (size_t)k + 0];
    double ky = kvecs[3 * (size_t)k + 1];
    double kz = kvecs[3 * (size_t)k + 2];

    double kr = gx * kx + gy * ky + gz * kz;

    float s, c;
    if (fp64) {
        s = (float)sin(kr);
        c = (float)cos(kr);
    } else {
        __sincosf((float)kr, &s, &c);
    }
    grid_c[(size_t)k * (size_t)M + (size_t)m] = c;
    grid_s[(size_t)k * (size_t)M + (size_t)m] = s;
}

__global__ void compute_atom_trig_kernel_f32(
    const double* __restrict__ atom_xyz, // [N*3]
    int N,
    const double* __restrict__ kvecs,    // [K*3]
    int K,
    int fp64,
    float* __restrict__ atom_c,          // [K*N]
    float* __restrict__ atom_s           // [K*N]
) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || k >= K) return;

    double ax = atom_xyz[3 * (size_t)a + 0];
    double ay = atom_xyz[3 * (size_t)a + 1];
    double az = atom_xyz[3 * (size_t)a + 2];

    double kx = kvecs[3 * (size_t)k + 0];
    double ky = kvecs[3 * (size_t)k + 1];
    double kz = kvecs[3 * (size_t)k + 2];

    double kr = ax * kx + ay * ky + az * kz;

    float s, c;
    if (fp64) {
        s = (float)sin(kr);
        c = (float)cos(kr);
    } else {
        __sincosf((float)kr, &s, &c);
    }
    atom_c[(size_t)k * (size_t)N + (size_t)a] = c;
    atom_s[(size_t)k * (size_t)N + (size_t)a] = s;
}

// ============================= Utils kernels =================================

__global__ void fill_ones_kernel(double* ones, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ones[i] = 1.0;
}

// ============================= Phi kernel (tiled) ============================
//
// Computes P (M x N), column-major with lda=M: P[m + a*M] = phi(m,a)
//
// Launch layout:
//   blockIdx.y = atom index a
//   blockIdx.x = m-tile index
//
// Threads:
//   x dimension iterates m within tile (blockDim.x)
//   y dimension iterates k within tile (blockDim.y == KT)
//
// Shared memory:
//   - atom_c/s for the current atom and current k-tile (KT)
//   - grid_c/s for current m-tile and current k-tile (KT * MT)
//
// This reduces redundant global loads significantly.

template<int MT, int KT>
__global__ void compute_phi_tiled_kernel(
    const double* __restrict__ grid_xyz, // [M*3]
    int M,
    const double* __restrict__ atom_xyz, // [N*3]
    int N,
    int fit_flag,
    // k-space trig cache (float)
    const float* __restrict__ grid_c,    // [K*M]
    const float* __restrict__ grid_s,    // [K*M]
    const float* __restrict__ atom_c,    // [K*N]
    const float* __restrict__ atom_s,    // [K*N]
    const double* __restrict__ kcoefs,   // [K]
    int K,
    // real-space
    const double* __restrict__ shifts,   // [S*3]
    int S,
    double sqrt_alpha,
    double R2,
    double self_term,
    // output
    double* __restrict__ P               // [M*N], lda=M (column-major)
) {
    int a = blockIdx.y;
    int m0 = blockIdx.x * MT;

    int tx = threadIdx.x; // [0..MT-1]
    int ty = threadIdx.y; // [0..KT-1]

    int m = m0 + tx;
    if (a >= N) return;

    double gx=0.0, gy=0.0, gz=0.0;
    if (m < M) {
        gx = grid_xyz[3 * (size_t)m + 0];
        gy = grid_xyz[3 * (size_t)m + 1];
        gz = grid_xyz[3 * (size_t)m + 2];
    }

    double ax = atom_xyz[3 * (size_t)a + 0];
    double ay = atom_xyz[3 * (size_t)a + 1];
    double az = atom_xyz[3 * (size_t)a + 2];

    double phi = 0.0;

    if (fit_flag == 0) {
        if (m < M) {
            double dx = gx - ax, dy = gy - ay, dz = gz - az;
            double r2 = dx*dx + dy*dy + dz*dz;
            phi = (r2 > 1e-24) ? rsqrt(r2) : 0.0;
            P[(size_t)m + (size_t)a * (size_t)M] = phi;
        }
        return;
    }

    extern __shared__ unsigned char smem[];
    // Layout:
    // atomC[KT], atomS[KT], gridC[KT*MT], gridS[KT*MT]
    float* s_atomC = reinterpret_cast<float*>(smem);
    float* s_atomS = s_atomC + KT;
    float* s_gridC = s_atomS + KT;
    float* s_gridS = s_gridC + KT * MT;

    // k-space sum
    double phi_recp = 0.0;

    for (int k0 = 0; k0 < K; k0 += KT) {
        int k = k0 + ty;

        // Load atom trig for this atom and k
        if (tx == 0) {
            if (k < K) {
                s_atomC[ty] = atom_c[(size_t)k * (size_t)N + (size_t)a];
                s_atomS[ty] = atom_s[(size_t)k * (size_t)N + (size_t)a];
            } else {
                s_atomC[ty] = 0.0f;
                s_atomS[ty] = 0.0f;
            }
        }

        // Load grid trig for this m-tile and k
        if (m < M) {
            if (k < K) {
                s_gridC[ty * MT + tx] = grid_c[(size_t)k * (size_t)M + (size_t)m];
                s_gridS[ty * MT + tx] = grid_s[(size_t)k * (size_t)M + (size_t)m];
            } else {
                s_gridC[ty * MT + tx] = 0.0f;
                s_gridS[ty * MT + tx] = 0.0f;
            }
        } else {
            // out-of-range m: keep shared values defined
            s_gridC[ty * MT + tx] = 0.0f;
            s_gridS[ty * MT + tx] = 0.0f;
        }

        __syncthreads();

        if (m < M) {
            // Each thread computes partial sum over its ty lane; then reduce over KT.
            // We use warp-level reduction; KT is set to 8/16/32 (here 8 default below).
            double partial = 0.0;
            if (k < K) {
                double cg = (double)s_gridC[ty * MT + tx];
                double sg = (double)s_gridS[ty * MT + tx];
                double ca = (double)s_atomC[ty];
                double sa = (double)s_atomS[ty];
                partial = (cg * ca + sg * sa) * kcoefs[(size_t)k];
            }

            // Reduce partial over ty dimension inside the block for fixed tx.
            // Assumes blockDim.y == KT and KT <= 32.
            // Use shared memory reduction on s_gridC buffer as scratch (safe because we sync).
            // We'll store partials into s_gridC[ty*MT+tx] as float? No. Use double scratch separately.
        }

        __syncthreads();

        // Do a simple reduction using shared double scratch per (ty, tx).
        // Allocate scratch in registers then use shared memory via reinterpret cast to double.
        // We can't alias float shared to double safely; instead, do two-stage: accumulate via atomicAdd? not ok.
        // Better: each thread directly accumulates across ty by iterating local KT since KT is small.
        // That would re-read shared but cheap.
        if (m < M) {
            double sumk = 0.0;
            #pragma unroll
            for (int kk = 0; kk < KT; kk++) {
                int kx = k0 + kk;
                if (kx < K) {
                    double cg = (double)s_gridC[kk * MT + tx];
                    double sg = (double)s_gridS[kk * MT + tx];
                    double ca = (double)s_atomC[kk];
                    double sa = (double)s_atomS[kk];
                    sumk += (cg * ca + sg * sa) * kcoefs[(size_t)kx];
                }
            }
            phi_recp += sumk;
        }

        __syncthreads();
    }

    // real-space sum over pruned shifts (S is small ~39)
    double phi_real = 0.0;
    if (m < M) {
        for (int si = 0; si < S; si++) {
            double sx = shifts[3 * (size_t)si + 0];
            double sy = shifts[3 * (size_t)si + 1];
            double sz = shifts[3 * (size_t)si + 2];

            double dx = gx - (ax + sx);
            double dy = gy - (ay + sy);
            double dz = gz - (az + sz);

            double rr2 = dx*dx + dy*dy + dz*dz;
            if (rr2 > R2) continue;

            if ((sx*sx + sy*sy + sz*sz) < 1e-28 && rr2 < 1e-24) continue;

            double rr = sqrt(rr2);
            phi_real += erfc(sqrt_alpha * rr) / rr;
        }

        phi = (phi_real + phi_recp) - self_term;
        P[(size_t)m + (size_t)a * (size_t)M] = phi;
    }
}

// ============================= Main =========================================

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank=0, size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Args args = parse_args(argc, argv);
    int gpu = select_gpu(rank, args.gpu_id);

    if (rank == 0) {
        std::cout << "===================================================================\n";
        std::cout << "REPEAT  MPI+CUDA OPT (Chunk Phi + cuBLAS + Tiled Phi)\n";
        std::cout << "===================================================================\n";
        std::cout << "Cube: " << args.cube_file << "\n";
        std::cout << "MPI ranks: " << size << "\n";
        std::cout << "Trig precision: " << (args.fp64 ? "fp64" : "fp32(default)") << "\n";
    }
    MPI_PRINT(rank, "Rank " << rank << " using GPU " << gpu << "\n");

    CubeData cube;
    FilteredGrid filtered_all;

    if (rank == 0) {
        cube = read_cube(args.cube_file);
        filtered_all = filter_grid_rank0(cube, args.vdw_factor, args.vdw_max, rank);
    }

    int n_atoms = 0;
    if (rank == 0) n_atoms = cube.n_atoms;
    MPI_Bcast(&n_atoms, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> atom_index((size_t)n_atoms);
    std::vector<Vec3> atom_pos((size_t)n_atoms);
    if (rank == 0) { atom_index = cube.atom_index; atom_pos = cube.atom_pos; }
    MPI_Bcast(atom_index.data(), n_atoms, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(atom_pos.data(), n_atoms * (int)sizeof(Vec3), MPI_BYTE, 0, MPI_COMM_WORLD);

    Vec3 axis_vector[3];
    int n_grid[3] = {0,0,0};
    Vec3 origin;
    if (rank == 0) {
        axis_vector[0] = cube.axis_vector[0];
        axis_vector[1] = cube.axis_vector[1];
        axis_vector[2] = cube.axis_vector[2];
        n_grid[0] = cube.n_grid[0];
        n_grid[1] = cube.n_grid[1];
        n_grid[2] = cube.n_grid[2];
        origin = cube.origin;
    }
    MPI_Bcast(axis_vector, 3 * (int)sizeof(Vec3), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_grid, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&origin, (int)sizeof(Vec3), MPI_BYTE, 0, MPI_COMM_WORLD);

    Vec3 box_vectors[3];
    for (int i = 0; i < 3; i++) box_vectors[i] = axis_vector[i] * n_grid[i];

    EwaldData ewald = setup_ewald_pruned_aabb(box_vectors, args.R_cutoff, args.fit_flag, atom_pos, rank);
    int K = (int)ewald.kvecs.size();
    int S = (int)ewald.shifts.size();

    long long n_total_filtered = 0;
    if (rank == 0) n_total_filtered = (long long)filtered_all.positions.size();
    MPI_Bcast(&n_total_filtered, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displs;
    partition_counts((int)n_total_filtered, size, counts, displs);
    int my_count = counts[(size_t)rank];

    std::vector<Vec3> my_pos((size_t)my_count);
    std::vector<double> my_V((size_t)my_count);

    std::vector<int> counts_bytes((size_t)size), displs_bytes((size_t)size);
    for (int r = 0; r < size; r++) {
        counts_bytes[(size_t)r] = counts[(size_t)r] * (int)sizeof(Vec3);
        displs_bytes[(size_t)r] = displs[(size_t)r] * (int)sizeof(Vec3);
    }

    MPI_Scatterv(rank == 0 ? (void*)filtered_all.positions.data() : nullptr,
                 counts_bytes.data(), displs_bytes.data(), MPI_BYTE,
                 (void*)my_pos.data(), my_count * (int)sizeof(Vec3), MPI_BYTE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? (void*)filtered_all.V_pot.data() : nullptr,
                 counts.data(), displs.data(), MPI_DOUBLE,
                 my_V.data(), my_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Filtered points total: " << n_total_filtered << "\n";
        std::cout << "Ewald: fit=" << args.fit_flag << " K=" << K << " shifts=" << S
                  << " alpha=" << std::fixed << std::setprecision(6) << ewald.alpha << "\n";
        std::cout << "Chunk size: " << args.chunk_size << "\n";
    }

    // ================= GPU buffers =================
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");

    // Atom xyz (double)
    std::vector<double> h_atom_xyz((size_t)n_atoms * 3);
    for (int i = 0; i < n_atoms; i++) {
        h_atom_xyz[3 * (size_t)i + 0] = atom_pos[(size_t)i].x;
        h_atom_xyz[3 * (size_t)i + 1] = atom_pos[(size_t)i].y;
        h_atom_xyz[3 * (size_t)i + 2] = atom_pos[(size_t)i].z;
    }

    double* d_atom_xyz=nullptr;
    cuda_check(cudaMalloc(&d_atom_xyz, sizeof(double) * h_atom_xyz.size()), "cudaMalloc d_atom_xyz");
    cuda_check(cudaMemcpy(d_atom_xyz, h_atom_xyz.data(), sizeof(double) * h_atom_xyz.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy atom_xyz");

    // Ewald arrays
    double *d_kvecs=nullptr, *d_kcoefs=nullptr, *d_shifts=nullptr;
    if (args.fit_flag != 0) {
        std::vector<double> h_kvecs((size_t)K * 3);
        for (int i = 0; i < K; i++) {
            h_kvecs[3 * (size_t)i + 0] = ewald.kvecs[(size_t)i].x;
            h_kvecs[3 * (size_t)i + 1] = ewald.kvecs[(size_t)i].y;
            h_kvecs[3 * (size_t)i + 2] = ewald.kvecs[(size_t)i].z;
        }
        std::vector<double> h_shifts((size_t)S * 3);
        for (int i = 0; i < S; i++) {
            h_shifts[3 * (size_t)i + 0] = ewald.shifts[(size_t)i].x;
            h_shifts[3 * (size_t)i + 1] = ewald.shifts[(size_t)i].y;
            h_shifts[3 * (size_t)i + 2] = ewald.shifts[(size_t)i].z;
        }

        cuda_check(cudaMalloc(&d_kvecs, sizeof(double) * (size_t)K * 3), "cudaMalloc d_kvecs");
        cuda_check(cudaMalloc(&d_kcoefs, sizeof(double) * (size_t)K), "cudaMalloc d_kcoefs");
        cuda_check(cudaMalloc(&d_shifts, sizeof(double) * (size_t)S * 3), "cudaMalloc d_shifts");

        cuda_check(cudaMemcpy(d_kvecs, h_kvecs.data(), sizeof(double) * (size_t)K * 3, cudaMemcpyHostToDevice),
                   "cudaMemcpy kvecs");
        cuda_check(cudaMemcpy(d_kcoefs, ewald.kcoefs.data(), sizeof(double) * (size_t)K, cudaMemcpyHostToDevice),
                   "cudaMemcpy kcoefs");
        cuda_check(cudaMemcpy(d_shifts, h_shifts.data(), sizeof(double) * (size_t)S * 3, cudaMemcpyHostToDevice),
                   "cudaMemcpy shifts");
    }

    // Chunk buffers (double for xyz and V)
    int max_chunk = std::max(1, args.chunk_size);
    double *d_grid_xyz=nullptr, *d_V=nullptr;
    cuda_check(cudaMalloc(&d_grid_xyz, sizeof(double) * (size_t)max_chunk * 3), "cudaMalloc d_grid_xyz");
    cuda_check(cudaMalloc(&d_V, sizeof(double) * (size_t)max_chunk), "cudaMalloc d_V");

    // P buffer (M x N) column-major lda=M
    double* d_P=nullptr;
    cuda_check(cudaMalloc(&d_P, sizeof(double) * (size_t)n_atoms * (size_t)max_chunk), "cudaMalloc d_P");

    // ones vector
    double* d_ones=nullptr;
    cuda_check(cudaMalloc(&d_ones, sizeof(double) * (size_t)max_chunk), "cudaMalloc d_ones");

    // trig caches (float)
    float *d_grid_c=nullptr, *d_grid_s=nullptr, *d_atom_c=nullptr, *d_atom_s=nullptr;
    if (args.fit_flag != 0) {
        cuda_check(cudaMalloc(&d_grid_c, sizeof(float) * (size_t)K * (size_t)max_chunk), "cudaMalloc d_grid_c");
        cuda_check(cudaMalloc(&d_grid_s, sizeof(float) * (size_t)K * (size_t)max_chunk), "cudaMalloc d_grid_s");
        cuda_check(cudaMalloc(&d_atom_c, sizeof(float) * (size_t)K * (size_t)n_atoms), "cudaMalloc d_atom_c");
        cuda_check(cudaMalloc(&d_atom_s, sizeof(float) * (size_t)K * (size_t)n_atoms), "cudaMalloc d_atom_s");

        dim3 bta(32, 8, 1);
        dim3 gta((n_atoms + bta.x - 1) / bta.x, (K + bta.y - 1) / bta.y, 1);
        compute_atom_trig_kernel_f32<<<gta, bta>>>(d_atom_xyz, n_atoms, d_kvecs, K, args.fp64, d_atom_c, d_atom_s);
        cuda_check(cudaGetLastError(), "compute_atom_trig_kernel_f32");
        cuda_check(cudaDeviceSynchronize(), "atom trig sync");
    }

    // Accumulators on device
    double *d_ATA=nullptr, *d_ATb=nullptr, *d_sum_phi=nullptr;
    cuda_check(cudaMalloc(&d_ATA, sizeof(double) * (size_t)n_atoms * (size_t)n_atoms), "cudaMalloc d_ATA");
    cuda_check(cudaMalloc(&d_ATb, sizeof(double) * (size_t)n_atoms), "cudaMalloc d_ATb");
    cuda_check(cudaMalloc(&d_sum_phi, sizeof(double) * (size_t)n_atoms), "cudaMalloc d_sum_phi");
    cuda_check(cudaMemset(d_ATA, 0, sizeof(double) * (size_t)n_atoms * (size_t)n_atoms), "memset ATA");
    cuda_check(cudaMemset(d_ATb, 0, sizeof(double) * (size_t)n_atoms), "memset ATb");
    cuda_check(cudaMemset(d_sum_phi, 0, sizeof(double) * (size_t)n_atoms), "memset sum_phi");

    std::vector<double> h_grid_xyz((size_t)max_chunk * 3);
    double sum_V_local = 0.0;

    const double R2 = args.R_cutoff * args.R_cutoff;
    const double sqrt_alpha = ewald.sqrt_alpha;
    const double self_term = ewald.self_term;

    const double one = 1.0;

    // ================= Main loop =================
    for (int off = 0; off < my_count; off += max_chunk) {
        int M = std::min(max_chunk, my_count - off);

        for (int i = 0; i < M; i++) {
            const Vec3& p = my_pos[(size_t)(off + i)];
            h_grid_xyz[3 * (size_t)i + 0] = p.x;
            h_grid_xyz[3 * (size_t)i + 1] = p.y;
            h_grid_xyz[3 * (size_t)i + 2] = p.z;
            sum_V_local += my_V[(size_t)(off + i)];
        }

        cuda_check(cudaMemcpy(d_grid_xyz, h_grid_xyz.data(), sizeof(double) * (size_t)M * 3, cudaMemcpyHostToDevice),
                   "cpy grid_xyz");
        cuda_check(cudaMemcpy(d_V, my_V.data() + off, sizeof(double) * (size_t)M, cudaMemcpyHostToDevice),
                   "cpy V");

        // ones[0..M-1] = 1
        {
            int threads = 256;
            int blocks = (M + threads - 1) / threads;
            fill_ones_kernel<<<blocks, threads>>>(d_ones, M);
            cuda_check(cudaGetLastError(), "fill_ones_kernel");
        }

        if (args.fit_flag != 0) {
            dim3 btg(32, 8, 1);
            dim3 gtg((M + btg.x - 1) / btg.x, (K + btg.y - 1) / btg.y, 1);
            compute_grid_trig_kernel_f32<<<gtg, btg>>>(d_grid_xyz, M, d_kvecs, K, args.fp64, d_grid_c, d_grid_s);
            cuda_check(cudaGetLastError(), "compute_grid_trig_kernel_f32");
        }

        // P kernel (tiled)
        // Choose MT/KT:
        //   MT: m tile size per block in x
        //   KT: k tile size (threadIdx.y)
        // Here MT=128, KT=8 => 1024 threads/block (max), good occupancy on many GPUs.
        // Shared size = 2*KT + 2*KT*MT floats.
        //           = 2*8 + 2*8*128 = 2064 floats = 8256 bytes.
        {
            constexpr int MT = 128;
            constexpr int KT = 8;
            dim3 block(MT, KT, 1);
            dim3 grid((M + MT - 1) / MT, n_atoms, 1);
            size_t shmem = (size_t)(2 * KT + 2 * KT * MT) * sizeof(float);

            compute_phi_tiled_kernel<MT, KT><<<grid, block, shmem>>>(
                d_grid_xyz, M,
                d_atom_xyz, n_atoms,
                args.fit_flag,
                d_grid_c, d_grid_s,
                d_atom_c, d_atom_s,
                d_kcoefs, K,
                d_shifts, S,
                sqrt_alpha, R2, self_term,
                d_P
            );
            cuda_check(cudaGetLastError(), "compute_phi_tiled_kernel launch");
        }

        // cuBLAS: treat d_P as (M x N) with lda=M
        // ATA += P^T * P  => (N x N)
        cublas_check(
            cublasDgemm(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n_atoms, n_atoms, M,
                &one,
                d_P, M,
                d_P, M,
                &one,
                d_ATA, n_atoms
            ),
            "cublasDgemm ATA"
        );

        // ATb += P^T * V  => (N)
        cublas_check(
            cublasDgemv(
                handle,
                CUBLAS_OP_T,
                M, n_atoms,
                &one,
                d_P, M,
                d_V, 1,
                &one,
                d_ATb, 1
            ),
            "cublasDgemv ATb"
        );

        // sum_phi += P^T * ones => (N)
        cublas_check(
            cublasDgemv(
                handle,
                CUBLAS_OP_T,
                M, n_atoms,
                &one,
                d_P, M,
                d_ones, 1,
                &one,
                d_sum_phi, 1
            ),
            "cublasDgemv sum_phi"
        );

        cuda_check(cudaDeviceSynchronize(), "chunk sync");
    }

    // Copy local accumulators to host
    std::vector<double> ATA_local((size_t)n_atoms * (size_t)n_atoms, 0.0);
    std::vector<double> ATb_local((size_t)n_atoms, 0.0);
    std::vector<double> sum_phi_local((size_t)n_atoms, 0.0);

    cuda_check(cudaMemcpy(ATA_local.data(), d_ATA, sizeof(double) * ATA_local.size(), cudaMemcpyDeviceToHost),
               "cpy ATA_local");
    cuda_check(cudaMemcpy(ATb_local.data(), d_ATb, sizeof(double) * ATb_local.size(), cudaMemcpyDeviceToHost),
               "cpy ATb_local");
    cuda_check(cudaMemcpy(sum_phi_local.data(), d_sum_phi, sizeof(double) * sum_phi_local.size(), cudaMemcpyDeviceToHost),
               "cpy sum_phi_local");

    // Reduce to rank0
    std::vector<double> ATA((size_t)n_atoms * (size_t)n_atoms, 0.0);
    std::vector<double> ATb((size_t)n_atoms, 0.0);
    std::vector<double> sum_phi((size_t)n_atoms, 0.0);
    double sum_V = 0.0;

    MPI_Reduce(ATA_local.data(), ATA.data(), (int)ATA.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(ATb_local.data(), ATb.data(), (int)ATb.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(sum_phi_local.data(), sum_phi.data(), (int)sum_phi.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_V_local, &sum_V, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_atom_xyz);
    if (d_kvecs) cudaFree(d_kvecs);
    if (d_kcoefs) cudaFree(d_kcoefs);
    if (d_shifts) cudaFree(d_shifts);

    cudaFree(d_grid_xyz);
    cudaFree(d_V);
    cudaFree(d_P);
    cudaFree(d_ones);

    if (d_grid_c) cudaFree(d_grid_c);
    if (d_grid_s) cudaFree(d_grid_s);
    if (d_atom_c) cudaFree(d_atom_c);
    if (d_atom_s) cudaFree(d_atom_s);

    cudaFree(d_ATA);
    cudaFree(d_ATb);
    cudaFree(d_sum_phi);

    // Rank0 solve
    if (rank == 0) {
        const long long Ngrid = n_total_filtered;
        const double invN = (Ngrid > 0) ? (1.0 / (double)Ngrid) : 0.0;

        std::vector<double> phi_bar((size_t)n_atoms, 0.0);
        for (int i = 0; i < n_atoms; i++) phi_bar[(size_t)i] = sum_phi[(size_t)i] * invN;
        const double V_bar = sum_V * invN;

        std::vector<std::vector<double>> A((size_t)n_atoms, std::vector<double>((size_t)n_atoms, 0.0));
        std::vector<double> b((size_t)n_atoms, 0.0);

        for (int i = 0; i < n_atoms; i++) {
            for (int j = 0; j < n_atoms; j++) {
                A[(size_t)i][(size_t)j] =
                    ATA[(size_t)i * (size_t)n_atoms + (size_t)j] - (double)Ngrid * phi_bar[(size_t)i] * phi_bar[(size_t)j];
            }
            b[(size_t)i] = ATb[(size_t)i] - (double)Ngrid * phi_bar[(size_t)i] * V_bar;
        }

        const int n = n_atoms + 1;
        std::vector<std::vector<double>> A_solv((size_t)n, std::vector<double>((size_t)n, 0.0));
        std::vector<double> b_solv((size_t)n, 0.0);

        for (int i = 0; i < n_atoms; i++) {
            for (int j = 0; j < n_atoms; j++) A_solv[(size_t)i][(size_t)j] = A[(size_t)i][(size_t)j];
            A_solv[(size_t)i][(size_t)n_atoms] = 1.0;
            A_solv[(size_t)n_atoms][(size_t)i] = 1.0;
            b_solv[(size_t)i] = b[(size_t)i];
        }
        b_solv[(size_t)n_atoms] = args.q_tot;

        std::vector<double> sol = solve_linear_system(A_solv, b_solv);
        std::vector<double> charges(sol.begin(), sol.begin() + n_atoms);

        std::cout << "Fitted charges ordered as within the cube file\n";
        std::cout << "----------------------------------------------\n";
        for (int i = 0; i < n_atoms; i++) {
            int atom_num = atom_index[(size_t)i];
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