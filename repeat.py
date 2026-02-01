#!/usr/bin/env python3
"""
REPEAT - Electrostatic Potential fitted charges for periodic systems
Optimized for multi-core performance with noGIL support
"""

import numpy as np
from scipy import special, linalg
import sys
import argparse
import time
import os
import itertools
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import threading
import warnings

try:
    import psutil
except ImportError:
    psutil = None

PI = np.pi
BOHR_TO_ANG = 0.529177

# -------------------- Performance Monitoring --------------------
class PerformanceMonitor:
    """Simple performance monitor for tracking execution time and memory usage."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
        self.start_times = {}
        
    def start(self, name):
        """Start timing a section."""
        if self.enabled:
            self.start_times[name] = time.time()
    
    def end(self, name):
        """End timing a section and record duration."""
        if self.enabled and name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            return duration
        return 0.0
    
    def report(self):
        """Print performance report."""
        if not self.enabled or not self.timings:
            return
            
        print("\n" + "="*60)
        print("Performance Report")
        print("="*60)
        print(f"{'Section':<30} {'Total (s)':<12} {'Avg (s)':<12} {'Calls':<8}")
        print("-"*60)
        
        for name in sorted(self.timings.keys()):
            times = self.timings[name]
            total = sum(times)
            avg = total / len(times)
            print(f"{name:<30} {total:<12.3f} {avg:<12.3f} {len(times):<8}")
        
        print("="*60)


# Global performance monitor
_perf_monitor = PerformanceMonitor(enabled=os.environ.get('REPEAT_PERF', '0') == '1')

# -------------------- Parallel Strategy Selection --------------------
# Thresholds for choosing parallel strategy
PARALLEL_STRATEGY_THRESHOLDS = {
    'numba_parallel_min_grid': int(os.environ.get('REPEAT_NUMBA_PARALLEL_MIN', '10000')),
    'gpu_min_grid': int(os.environ.get('REPEAT_GPU_MIN', '5000')),  # Minimum grid for GPU
}

# Global GPU configuration (set by command line)
_GPU_CONFIG = {
    'enabled': False,
    'devices': [0],  # Default to GPU 0
    'use_mixed_precision': False,  # fp32 for speed, fp64 for accuracy
}

def parse_gpu_devices(gpu_arg):
    """Parse GPU device IDs from command line argument.
    
    Args:
        gpu_arg: None, '0', '0,1,2', or 'all'
    
    Returns:
        list: List of GPU device IDs
    """
    if gpu_arg is None or gpu_arg == '':
        return [0] if _GPU_INFO['device_count'] > 0 else []
    
    if gpu_arg.lower() == 'all':
        return list(range(_GPU_INFO['device_count']))
    
    try:
        devices = [int(x.strip()) for x in str(gpu_arg).split(',')]
        # Validate devices
        valid_devices = [d for d in devices if 0 <= d < _GPU_INFO['device_count']]
        if len(valid_devices) != len(devices):
            invalid = set(devices) - set(valid_devices)
            print(f"Warning: Invalid GPU IDs {invalid}, using {valid_devices}")
        return valid_devices if valid_devices else [0]
    except ValueError:
        print(f"Warning: Invalid GPU argument '{gpu_arg}', using GPU 0")
        return [0]

def set_gpu_config(enabled=True, devices=None, mixed_precision=False):
    """Configure GPU acceleration.
    
    Args:
        enabled: Whether to enable GPU
        devices: List of GPU device IDs or None for auto
        mixed_precision: Whether to use fp32 (faster) instead of fp64
    """
    global _GPU_CONFIG
    
    if not _GPU_INFO['available']:
        _GPU_CONFIG['enabled'] = False
        return False
    
    _GPU_CONFIG['enabled'] = enabled
    
    if devices is not None:
        _GPU_CONFIG['devices'] = devices
    else:
        _GPU_CONFIG['devices'] = [0]
    
    _GPU_CONFIG['use_mixed_precision'] = mixed_precision
    
    if enabled:
        device_info = []
        total_mem = 0
        for d in _GPU_CONFIG['devices']:
            if _GPU_INFO['cupy_available']:
                import cupy as cp
                with cp.cuda.Device(d):
                    props = cp.cuda.runtime.getDeviceProperties(d)
                    name = props['name'].decode('utf-8')
                    # Get memory info for this device
                    free_mem, total_mem_gb = cp.cuda.Device(d).mem_info
                    total_mem_gb = total_mem_gb / (1024**3)
                    free_mem_gb = free_mem / (1024**3)
                    device_info.append(f"GPU{d}:{name} ({free_mem_gb:.1f}GB/{total_mem_gb:.1f}GB free)")
                    total_mem += free_mem
            else:
                device_info.append(f"GPU{d}")
        
        precision = "mixed (fp32)" if mixed_precision else "fp64"
        print(f"GPU acceleration enabled: {', '.join(device_info)}")
        print(f"  Devices: {_GPU_CONFIG['devices']}")
        print(f"  Precision: {precision}")
        if total_mem > 0:
            print(f"  Total available GPU memory: {total_mem / (1024**3):.1f} GB")
    
    return True

def select_parallel_strategy(n_grid, n_atoms, use_jit=True, force_strategy=None, prefer_gpu=None):
    """
    Select optimal parallel strategy based on system size and available backends.
    
    Strategies (in order of preference):
    1. 'gpu' - GPU acceleration (CuPy/PyTorch) - supports multi-GPU
    2. 'numba_parallel' - Numba prange (single process, internal parallelism)
    3. 'nogil_threads' - noGIL Python threading (shared memory, no serialization)
    4. 'multiprocess' - Standard multiprocessing (process isolation)
    5. 'sequential' - Single-threaded (fallback)
    
    Args:
        n_grid: Number of grid points
        n_atoms: Number of atoms
        use_jit: Whether JIT is available
        force_strategy: Force specific strategy (for testing)
        prefer_gpu: Whether to prefer GPU (if None, uses global _GPU_CONFIG['enabled'])
        
    Returns:
        tuple: (strategy_name, n_cores, use_parallel_jit, n_gpus)
    """
    # Determine if GPU should be used
    use_gpu = prefer_gpu if prefer_gpu is not None else _GPU_CONFIG['enabled']
    
    if force_strategy:
        if force_strategy == 'gpu':
            n_gpus = len(_GPU_CONFIG['devices']) if _GPU_CONFIG['enabled'] else 1
            return 'gpu', 1, False, n_gpus
        elif force_strategy == 'numba_parallel':
            return 'numba_parallel', 1, True, 0
        elif force_strategy == 'nogil_threads':
            return 'nogil_threads', _get_optimal_workers(), use_jit and _JIT_OK, 0
        elif force_strategy == 'multiprocess':
            return 'multiprocess', _get_optimal_workers(), use_jit and _JIT_OK, 0
        elif force_strategy == 'sequential':
            return 'sequential', 1, use_jit and _JIT_OK, 0
    
    # Check GPU availability and size threshold
    if use_gpu and _GPU_INFO['available'] and n_grid >= PARALLEL_STRATEGY_THRESHOLDS['gpu_min_grid']:
        n_gpus = len(_GPU_CONFIG['devices'])
        return 'gpu', 1, False, n_gpus
    
    # Check Numba parallel for large grids
    if use_jit and _JIT_PARALLEL_OK and n_grid >= PARALLEL_STRATEGY_THRESHOLDS['numba_parallel_min_grid']:
        return 'numba_parallel', 1, True, 0
    
    # Check if work is too small for any parallelism
    total_work = n_grid * n_atoms
    if total_work < 1e5:  # Only skip parallelism for very small systems
        return 'sequential', 1, use_jit and _JIT_OK, 0
    
    # noGIL Python: use threading instead of multiprocessing (shared memory advantage)
    if _NOGIL_ENABLED:
        return 'nogil_threads', _get_optimal_workers(), use_jit and _JIT_OK, 0
    
    # Standard Python: use multiprocessing for medium systems
    return 'multiprocess', _get_optimal_workers(), use_jit and _JIT_OK, 0

# -------------------- optional Numba JIT --------------------
try:
    from numba import njit  # type: ignore
    _NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    _NUMBA_AVAILABLE = False

# -------------------- CuPy GPU Detection --------------------
def _detect_gpu():
    """Detect GPU availability and return info."""
    gpu_info = {
        'available': False,
        'device_count': 0,
        'device_name': None,
        'cupy_available': False,
        'torch_available': False,
    }
    
    # Try CuPy first (preferred for NumPy compatibility)
    try:
        import cupy as cp
        gpu_info['cupy_available'] = True
        gpu_info['device_count'] = cp.cuda.runtime.getDeviceCount()
        if gpu_info['device_count'] > 0:
            gpu_info['available'] = True
            # Get first device name
            with cp.cuda.Device(0):
                props = cp.cuda.runtime.getDeviceProperties(0)
                gpu_info['device_name'] = props['name'].decode('utf-8')
    except Exception:
        pass
    
    # Try PyTorch as fallback
    if not gpu_info['available']:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['torch_available'] = True
                gpu_info['available'] = True
                gpu_info['device_count'] = torch.cuda.device_count()
                gpu_info['device_name'] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    return gpu_info

_GPU_INFO = _detect_gpu()

# -------------------- noGIL Detection --------------------
def _is_nogil():
    """Detect if running on Python with noGIL support (3.13+ experimental)."""
    try:
        # Method 1: Check sys.flags for nogil flag (Python 3.13+)
        if hasattr(sys.flags, 'nogil') and sys.flags.nogil:
            return True
        
        # Method 2: Check if threading can run without GIL
        # In noGIL Python, we can check the _is_gil_enabled function
        if hasattr(sys, '_is_gil_enabled'):
            return not sys._is_gil_enabled()
        
        # Method 3: Try to import experimental nogil features
        try:
            import sysconfig
            config_vars = sysconfig.get_config_vars()
            if 'Py_GIL_DISABLED' in config_vars:
                return bool(config_vars['Py_GIL_DISABLED'])
        except:
            pass
            
    except Exception:
        pass
    return False

_NOGIL_ENABLED = _is_nogil()

# -------------------- Parallel Backend Selection --------------------
def _pick_parallel_backend(prefer_threads=None):
    """
    Select optimal parallel backend based on Python GIL status.
    
    Returns:
        tuple: (executor_class, use_threads, context)
        - executor_class: ThreadPoolExecutor or ProcessPoolExecutor
        - use_threads: bool indicating if using thread-based parallelism
        - context: multiprocessing context (for process-based)
    """
    if prefer_threads is None:
        # Auto-detect: use threads if noGIL is enabled
        prefer_threads = _NOGIL_ENABLED
    
    if prefer_threads:
        # Use ThreadPoolExecutor for noGIL Python
        return ThreadPoolExecutor, True, None
    else:
        # Use ProcessPoolExecutor for standard Python
        from multiprocessing import get_context
        try:
            if sys.platform.startswith("linux"):
                ctx = get_context("forkserver")
            else:
                ctx = get_context("spawn")
        except Exception:
            ctx = get_context("spawn")
        
        # Return a wrapper that uses the context
        def ContextPoolExecutor(max_workers=None):
            return ctx.Pool(processes=max_workers)
        
        return ContextPoolExecutor, False, ctx


def _get_optimal_workers(n_cores_requested=None, reserve_fraction=0.1):
    """
    Determine optimal number of workers based on system and workload.
    
    For noGIL: Can use more threads since no GIL contention
    For standard Python: Reserve some cores for system processes
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 4
    
    total_cpus = max(1, cpu_count)
    
    if n_cores_requested is not None:
        return max(1, min(int(n_cores_requested), total_cpus))
    
    if _NOGIL_ENABLED:
        # In noGIL mode, we can use all cores efficiently
        # Reserve only a small fraction for system
        return max(1, int(total_cpus * (1 - reserve_fraction)))
    else:
        # Standard Python with GIL - use most cores
        # Only reserve 1 core for system on small systems
        if total_cpus <= 2:
            return total_cpus
        elif total_cpus <= 4:
            return max(1, total_cpus - 1)
        else:
            # Use 90% of cores for large systems
            return max(1, int(total_cpus * 0.9))

# -------------------- utility functions --------------------

def determine_optimal_cores(n_atoms=None, n_grid=None, n_cores_requested=None, reserve_fraction=0.25):
    """确定最佳 CPU 核心数。"""
    # Handle None from os.cpu_count()
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 4  # Default fallback
    
    total_cpus = max(1, cpu_count // 2)
    if n_cores_requested is not None:
        return max(1, min(int(n_cores_requested), total_cpus))

    max_cores = total_cpus
    if total_cpus > 4:
        max_cores = max(1, int(total_cpus * (1 - reserve_fraction)))

    if n_atoms is not None and n_grid is not None:
        work_per_core = (n_atoms * n_grid) / max_cores
        if work_per_core < 1e6:
            return max(1, min(4, max_cores // 4))
        elif work_per_core < 1e7:
            return max(1, min(16, max_cores // 2))
    return max(1, max_cores)

VDW_RADII = np.array([
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
], dtype=np.float64)

ATOM_SYMBOLS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
                "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
                "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
                "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
                "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
                "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
                "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
                "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

# QEq default parameters in Hartree atomic units (a.u.)
# Format: (electronegativity, hardness=J_0/2)
# 
# IMPORTANT: These parameters are in Hartree atomic units to match the C++ version.
# The "hardness" stored here is actually J_0/2 (half the idempotential), as used
# in Materials Studio and the original REPEAT code.
#
# Reference: Rappe and Goddard (1991) J. Phys. Chem. 95, 3358-3363
# To convert from eV to Hartree: divide by 27.211386245988
#
# Example conversions:
#   H:  chi=4.528 eV,  J_0/2=6.945 eV  ->  0.1664, 0.2552 (Hartree)
#   O:  chi=8.741 eV,  J_0/2=6.682 eV  ->  0.3212, 0.2456 (Hartree)
#
# In the QEq energy expression: E = chi*q + (1/2)*J_0*q^2
# Since hardness = J_0/2, we have: E = chi*q + hardness*q^2

DEFAULT_QEQ_PARAMS = {
    # Row 1-2: Light elements (H-Ne)
    "H": (0.166401, 0.255225), "He": (0.449802, 0.899603),
    "Li": (0.110466, 0.175364), "Be": (0.179222, 0.326547),
    "B": (0.149493, 0.310892), "C": (0.199590, 0.215240),
    "N": (0.245778, 0.243355), "O": (0.320234, 0.314869),
    "F": (0.235784, 0.409058), "Ne": (0.396149, 0.792298),
    # Row 3: Na-Ar
    "Na": (0.104476, 0.168749), "Mg": (0.145193, 0.271424),
    "Al": (0.111752, 0.216449), "Si": (0.153168, 0.256284),
    "P": (0.200757, 0.293988), "S": (0.254594, 0.329707),
    "Cl": (0.213918, 0.267278), "Ar": (0.289578, 0.579156),
    # Row 4: K-Ca + First transition series
    "K": (0.088968, 0.141114), "Ca": (0.118734, 0.211671),
    "Sc": (0.135014, 0.151110), "Ti": (0.142033, 0.166471),
    "V": (0.150415, 0.154972), "Cr": (0.136447, 0.223945),
    "Mn": (0.169778, 0.182273), "Fe": (0.152763, 0.229605),
    "Co": (0.156402, 0.233353), "Ni": (0.161252, 0.237542),
    "Cu": (0.199512, 0.127447), "Zn": (0.136010, 0.164010),
    "Ga": (0.110209, 0.220564), "Ge": (0.148868, 0.252683),
    "As": (0.190651, 0.279950), "Se": (0.236219, 0.303616),
    "Br": (0.209178, 0.321925), "Kr": (0.257239, 0.514479),
    # Row 5: Rb-Sr + Second transition series
    "Rb": (0.085661, 0.135675), "Sr": (0.111127, 0.179333),
    "Y": (0.144017, 0.165515), "Zr": (0.141408, 0.208143),
    "Nb": (0.141004, 0.216375), "Mo": (0.144164, 0.233500),
    "Tc": (0.153425, 0.236293), "Ru": (0.154711, 0.232250),
    "Rh": (0.158349, 0.233132), "Pd": (0.166618, 0.222475),
    "Ag": (0.163016, 0.230340), "Cd": (0.184992, 0.290828),
    "In": (0.110135, 0.205057), "Sn": (0.146516, 0.229605),
    "Sb": (0.180031, 0.245627), "Te": (0.213729, 0.259150),
    "I": (0.199586, 0.210207), "Xe": (0.222696, 0.445392),
    # Row 6: Cs-Ba + Lanthanides
    "Cs": (0.080222, 0.125753), "Ba": (0.103410, 0.176099),
    "La": (0.113553, 0.163898), "Ce": (0.113920, 0.161693),
    "Pr": (0.113920, 0.161693), "Nd": (0.113920, 0.161693),
    "Pm": (0.113920, 0.161693), "Sm": (0.113920, 0.161693),
    "Eu": (0.113920, 0.161693), "Gd": (0.113920, 0.161693),
    "Th": (0.113920, 0.161693),
}

# -------------------- memory-safe compute helpers --------------------

def _make_neighbor_shifts_27(box_vectors: np.ndarray) -> np.ndarray:
    shifts = []
    for kz in (-1, 0, 1):
        for ky in (-1, 0, 1):
            for kx in (-1, 0, 1):
                shifts.append(kx * box_vectors[0] + ky * box_vectors[1] + kz * box_vectors[2])
    return np.asarray(shifts, dtype=np.float64)

def _make_realspace_shifts(nmax: np.ndarray, box_vectors: np.ndarray) -> np.ndarray:
    bx, by, bz = box_vectors
    shifts = []
    for ix in range(-nmax[0], nmax[0] + 1):
        for iy in range(-nmax[1], nmax[1] + 1):
            for iz in range(-nmax[2], nmax[2] + 1):
                shifts.append(ix * bx + iy * by + iz * bz)
    return np.asarray(shifts, dtype=np.float64)

# ---- optional JIT kernels (probe + safe fallback) ----
_JIT_OK = False
if _NUMBA_AVAILABLE:
    try:
        import math

        @njit(cache=True, fastmath=True)
        def _reciprocal_sum_blocked_jit(delta, kvecs, kcoefs, block_k=64):
            n_grid = delta.shape[0]
            n_k = kvecs.shape[0]
            phi = np.zeros(n_grid, dtype=np.float64)
            for start in range(0, n_k, block_k):
                end = min(start + block_k, n_k)
                for i in range(n_grid):
                    acc = 0.0
                    dx = delta[i, 0]
                    dy = delta[i, 1]
                    dz = delta[i, 2]
                    for j in range(start, end):
                        kr = dx * kvecs[j, 0] + dy * kvecs[j, 1] + dz * kvecs[j, 2]
                        acc += math.cos(kr) * kcoefs[j]
                    phi[i] += acc
            return phi

        @njit(cache=True, fastmath=True)
        def _real_space_sum_jit(grid_pos, atom_pos, shifts, R_cutoff, sqrt_alpha):
            n_grid = grid_pos.shape[0]
            phi_real = np.zeros(n_grid, dtype=np.float64)
            R2 = R_cutoff * R_cutoff
            for s in range(shifts.shape[0]):
                sh0 = shifts[s, 0]
                sh1 = shifts[s, 1]
                sh2 = shifts[s, 2]
                img0 = atom_pos[0] + sh0
                img1 = atom_pos[1] + sh1
                img2 = atom_pos[2] + sh2

                is_self = (sh0 == 0.0 and sh1 == 0.0 and sh2 == 0.0)
                for i in range(n_grid):
                    dx = grid_pos[i, 0] - img0
                    dy = grid_pos[i, 1] - img1
                    dz = grid_pos[i, 2] - img2
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 <= R2:
                        if is_self and r2 <= 1e-20:
                            continue
                        r = math.sqrt(r2)
                        phi_real[i] += math.erfc(sqrt_alpha * r) / r
            return phi_real

        # Probe compile (disable JIT if erfc unsupported)
        _probe_grid = np.zeros((1, 3), dtype=np.float64)
        _probe_atom = np.zeros(3, dtype=np.float64)
        _probe_shifts = np.zeros((1, 3), dtype=np.float64)
        _ = _real_space_sum_jit(_probe_grid, _probe_atom, _probe_shifts, 1.0, 1.0)
        _ = _reciprocal_sum_blocked_jit(_probe_grid, _probe_shifts, np.zeros(1, dtype=np.float64), 1)
        _JIT_OK = True
        
        # Try to compile parallel versions
        try:
            from numba import prange
            
            @njit(cache=True, fastmath=True, parallel=True)
            def _reciprocal_sum_blocked_jit_parallel(delta, kvecs, kcoefs, block_k=64):
                """Parallel version using prange - for large grids."""
                n_grid = delta.shape[0]
                n_k = kvecs.shape[0]
                phi = np.zeros(n_grid, dtype=np.float64)
                for start in range(0, n_k, block_k):
                    end = min(start + block_k, n_k)
                    # Parallelize over grid points
                    for i in prange(n_grid):
                        acc = 0.0
                        dx = delta[i, 0]
                        dy = delta[i, 1]
                        dz = delta[i, 2]
                        for j in range(start, end):
                            kr = dx * kvecs[j, 0] + dy * kvecs[j, 1] + dz * kvecs[j, 2]
                            acc += math.cos(kr) * kcoefs[j]
                        phi[i] += acc
                return phi

            @njit(cache=True, fastmath=True, parallel=True)
            def _real_space_sum_jit_parallel(grid_pos, atom_pos, shifts, R_cutoff, sqrt_alpha):
                """Parallel version using prange - for large grids."""
                n_grid = grid_pos.shape[0]
                phi_real = np.zeros(n_grid, dtype=np.float64)
                R2 = R_cutoff * R_cutoff
                for s in range(shifts.shape[0]):
                    sh0 = shifts[s, 0]
                    sh1 = shifts[s, 1]
                    sh2 = shifts[s, 2]
                    img0 = atom_pos[0] + sh0
                    img1 = atom_pos[1] + sh1
                    img2 = atom_pos[2] + sh2

                    is_self = (sh0 == 0.0 and sh1 == 0.0 and sh2 == 0.0)
                    # Parallelize over grid points
                    for i in prange(n_grid):
                        dx = grid_pos[i, 0] - img0
                        dy = grid_pos[i, 1] - img1
                        dz = grid_pos[i, 2] - img2
                        r2 = dx * dx + dy * dy + dz * dz
                        if r2 <= R2:
                            if is_self and r2 <= 1e-20:
                                continue
                            r = math.sqrt(r2)
                            phi_real[i] += math.erfc(sqrt_alpha * r) / r
                return phi_real
            
            # Test parallel compilation
            _ = _real_space_sum_jit_parallel(_probe_grid, _probe_atom, _probe_shifts, 1.0, 1.0)
            _ = _reciprocal_sum_blocked_jit_parallel(_probe_grid, _probe_shifts, np.zeros(1, dtype=np.float64), 1)
            _JIT_PARALLEL_OK = True
        except Exception:
            _JIT_PARALLEL_OK = False
    except Exception:
        _JIT_OK = False
        _JIT_PARALLEL_OK = False

def _real_space_sum(grid_pos, atom_pos, shifts, R_cutoff, sqrt_alpha, use_jit=False):
    if use_jit and _JIT_OK:
        return _real_space_sum_jit(grid_pos, atom_pos, shifts, R_cutoff, sqrt_alpha)

    phi_real = np.zeros(grid_pos.shape[0], dtype=np.float64)
    # Pre-compute zero shift index for faster checking
    zero_shift_idx = -1
    for idx, sh in enumerate(shifts):
        if sh[0] == 0.0 and sh[1] == 0.0 and sh[2] == 0.0:
            zero_shift_idx = idx
            break
    
    for idx, sh in enumerate(shifts):
        img = atom_pos + sh
        d = np.linalg.norm(grid_pos - img, axis=1)
        # Use index comparison instead of array comparison
        if idx == zero_shift_idx:
            mask = (d > 1e-10) & (d <= R_cutoff)
        else:
            mask = (d <= R_cutoff)
        if np.any(mask):
            phi_real[mask] += special.erfc(sqrt_alpha * d[mask]) / d[mask]
    return phi_real

def _reciprocal_sum_blocked(delta, kvecs, kcoefs, block_k=64, use_jit=False):
    if use_jit and _JIT_OK:
        return _reciprocal_sum_blocked_jit(delta, kvecs, kcoefs, block_k=block_k)

    n_grid = delta.shape[0]
    phi = np.zeros(n_grid, dtype=np.float64)
    n_k = kvecs.shape[0]
    for start in range(0, n_k, block_k):
        end = min(start + block_k, n_k)
        kv = kvecs[start:end]
        kc = kcoefs[start:end]
        kr = delta @ kv.T
        phi += np.cos(kr) @ kc
    return phi

def compute_ewald_atom(args):
    """
    Worker function for Ewald potential of one atom on a set of grid points.
    """
    (atom_idx, atom_pos, grid_pos, kvecs, kcoefs, shifts, R_cutoff,
     sqrt_alpha, fit_flag, block_k, use_jit) = args

    delta = grid_pos - atom_pos

    if fit_flag == 0:
        dists = np.linalg.norm(delta, axis=1)
        phi = np.where(dists > 1e-10, 1.0 / dists, 0.0)
        return atom_idx, phi

    phi_recp = _reciprocal_sum_blocked(delta, kvecs, kcoefs, block_k=block_k, use_jit=use_jit)
    phi_real = _real_space_sum(grid_pos, atom_pos, shifts, R_cutoff, sqrt_alpha, use_jit=use_jit)
    return atom_idx, (phi_real + phi_recp)

def _compute_ewald_atom_worker(atom_idx, atom_pos, grid_pos, kvecs, kcoefs, shifts,
                                R_cutoff, sqrt_alpha, fit_flag, block_k, use_jit):
    """
    Worker function for computing Ewald potential of one atom.
    Designed to work with both threads (noGIL) and processes.
    """
    delta = grid_pos - atom_pos

    if fit_flag == 0:
        dists = np.linalg.norm(delta, axis=1)
        phi = np.where(dists > 1e-10, 1.0 / dists, 0.0)
        return atom_idx, phi

    phi_recp = _reciprocal_sum_blocked(delta, kvecs, kcoefs, block_k=block_k, use_jit=use_jit)
    phi_real = _real_space_sum(grid_pos, atom_pos, shifts, R_cutoff, sqrt_alpha, use_jit=use_jit)
    return atom_idx, (phi_real + phi_recp)

def _compute_ewald_batch_wrapper(args):
    """
    Batch wrapper for process-based parallelism.
    Processes multiple atoms in one go to reduce IPC overhead.
    """
    (start_idx, job_batch, grid_pos, kvecs, kcoefs, shifts,
     R_cutoff, sqrt_alpha, fit_flag, block_k, use_jit) = args
    
    results = []
    for atom_idx, atom_pos in job_batch:
        _, col = _compute_ewald_atom_worker(
            atom_idx, atom_pos, grid_pos, kvecs, kcoefs, shifts,
            R_cutoff, sqrt_alpha, fit_flag, block_k, use_jit
        )
        results.append((atom_idx, col))
    return results

class EwaldCalculator:
    """
    Ewald summation calculator.
    Supports memory-safe batch processing via `compute_batch_parallel`.
    """

    def __init__(self, box_vectors, volume, R_cutoff=20.0, fit_flag=1, n_cores=None, block_k=64, use_jit=True):
        self.box_vectors = np.asarray(box_vectors, dtype=np.float64)
        self.volume = float(volume)
        self.R_cutoff = float(R_cutoff)
        self.fit_flag = int(fit_flag)

        self.alpha = (PI / self.R_cutoff) ** 2
        self.sqrt_alpha = float(np.sqrt(self.alpha))

        self.block_k = int(block_k) if block_k is not None else 256
        self.use_jit = bool(use_jit and _JIT_OK)

        self.n_cores = determine_optimal_cores(n_cores_requested=n_cores)

        if fit_flag == 1:
            a, b, c = self.box_vectors
            self.recip_vectors = np.array([
                2 * PI * np.cross(b, c) / self.volume,
                2 * PI * np.cross(c, a) / self.volume,
                2 * PI * np.cross(a, b) / self.volume
            ], dtype=np.float64)

            self.nmax = np.array([
                int(np.floor(self.R_cutoff / np.linalg.norm(self.box_vectors[i]))) + 1
                for i in range(3)
            ], dtype=np.int64)

            self._setup_kspace()
            self.shifts = _make_realspace_shifts(self.nmax, self.box_vectors)
        else:
            self.recip_vectors = None
            self.nmax = np.array([0, 0, 0], dtype=np.int64)
            self.kvecs = np.empty((0, 3), dtype=np.float64)
            self.kcoefs = np.empty((0,), dtype=np.float64)
            self.shifts = np.zeros((1, 3), dtype=np.float64)

    def _setup_kspace(self):
        KMAX, KSQMAX = 7, 49
        beta = 1.0 / (4.0 * self.alpha)

        kvecs, kcoefs = [], []
        for kx in range(KMAX + 1):
            for ky in range(-KMAX, KMAX + 1):
                for kz in range(-KMAX, KMAX + 1):
                    ksq = kx * kx + ky * ky + kz * kz
                    if ksq >= KSQMAX or ksq == 0:
                        continue
                    kvec = kx * self.recip_vectors[0] + ky * self.recip_vectors[1] + kz * self.recip_vectors[2]
                    ksq_val = np.dot(kvec, kvec)
                    kvecs.append(kvec)
                    kcoefs.append((4.0 * PI / self.volume) * np.exp(-beta * ksq_val) / ksq_val)

        self.kvecs = np.asarray(kvecs, dtype=np.float64)
        self.kcoefs = np.asarray(kcoefs, dtype=np.float64)

    def compute_batch_parallel(self, grid_pos, atom_positions, n_cores=None, progress=True,
                               adaptive_block=True, min_block_k=8, max_retries=10):
        """
        Compute Ewald potential for a set of grid points and atoms.
        Returns a matrix of shape (n_grid, n_atoms).
        
        Automatically selects optimal parallel strategy based on system size:
        - Large grids (>=10k): Numba parallel (prange) - no process overhead
        - Medium grids: Multiprocessing with JIT
        - Small grids (<1M work): Sequential JIT
        """
        n_atoms = len(atom_positions)
        n_grid = len(grid_pos)
        
        # Select parallel strategy
        strategy, n_workers, use_jit, n_gpus = select_parallel_strategy(
            n_grid, n_atoms, self.use_jit
        )
        
        if progress:
            print(f"  Strategy: {strategy}, workers: {n_workers}, JIT: {use_jit}")
            if strategy == 'numba_parallel':
                print(f"  Using Numba parallel (grid size {n_grid} >= {PARALLEL_STRATEGY_THRESHOLDS['numba_parallel_min_grid']})")
            elif strategy == 'gpu':
                precision = "mixed" if _GPU_CONFIG['use_mixed_precision'] else "fp64"
                if n_gpus > 1:
                    print(f"  Using {n_gpus} GPUs ({precision} precision)")
                else:
                    print(f"  Using GPU {_GPU_CONFIG['devices'][0]} ({precision} precision)")
        
        # Execute with selected strategy
        if strategy == 'gpu':
            return self._compute_gpu(
                grid_pos, atom_positions, progress=progress
            )
        elif strategy == 'numba_parallel':
            return self._compute_numba_parallel(
                grid_pos, atom_positions, progress=progress
            )
        elif strategy == 'nogil_threads':
            # noGIL threading uses shared memory, similar to multiprocess but with threads
            return self._compute_nogil_threads(
                grid_pos, atom_positions, n_cores=n_workers, progress=progress
            )
        elif strategy == 'multiprocess':
            return self._compute_multiprocess(
                grid_pos, atom_positions, n_cores=n_workers, progress=progress,
                adaptive_block=adaptive_block, min_block_k=min_block_k, max_retries=max_retries
            )
        else:  # sequential
            return self._compute_sequential(
                grid_pos, atom_positions, use_jit=use_jit, progress=progress
            )
    
    def _compute_gpu(self, grid_pos, atom_positions, progress=True):
        """Compute using GPU acceleration with multi-GPU and mixed precision support."""
        import numpy as np
        
        n_grid = len(grid_pos)
        n_atoms = len(atom_positions)
        devices = _GPU_CONFIG['devices']
        n_gpus = len(devices)
        use_fp32 = _GPU_CONFIG['use_mixed_precision']
        
        dtype_str = "fp32" if use_fp32 else "fp64"
        if progress:
            if n_gpus > 1:
                print(f"  Computing {n_atoms} atoms on {n_gpus} GPUs ({dtype_str})...")
            else:
                print(f"  Computing {n_atoms} atoms on GPU {devices[0]} ({dtype_str})...")
        
        if _GPU_INFO['cupy_available']:
            return self._compute_gpu_cupy(grid_pos, atom_positions, devices, use_fp32, progress)
        elif _GPU_INFO['torch_available']:
            return self._compute_gpu_torch(grid_pos, atom_positions, devices, use_fp32, progress)
        else:
            raise RuntimeError("GPU requested but no GPU backend available")
    
    def _compute_gpu_cupy(self, grid_pos, atom_positions, devices, use_fp32, progress):
        """Compute using CuPy with optimized CPU-GPU communication and memory management.
        
        Optimizations:
        1. Batch transfer: Transfer all atoms at once instead of one by one
        2. Pinned memory: Use pinned memory for faster transfers
        3. Streams: Use CUDA streams for asynchronous operations
        4. Pre-allocation: Pre-allocate GPU memory to avoid repeated allocation
        5. Memory-aware batching: Automatically split computation to fit GPU memory
        """
        import cupy as cp
        from cupyx.scipy.special import erfc as cupy_erfc
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        n_grid = len(grid_pos)
        n_atoms = len(atom_positions)
        n_gpus = len(devices)
        
        # Choose dtype
        cp_dtype = cp.float32 if use_fp32 else cp.float64
        dtype_size = 4 if use_fp32 else 8
        
        if progress:
            print(f"    GPU: Checking memory requirements...")
        
        # Calculate memory requirements
        n_shifts = len(self.shifts)
        n_k = len(self.kvecs)
        
        # Constants (transferred once, stay in GPU memory)
        bytes_per_grid = n_grid * 3 * dtype_size
        bytes_per_shifts = n_shifts * 3 * dtype_size
        bytes_per_kvecs = n_k * 3 * dtype_size
        bytes_per_kcoefs = n_k * dtype_size
        constant_memory = bytes_per_grid + bytes_per_shifts + bytes_per_kvecs + bytes_per_kcoefs
        
        def estimate_batch_memory(n_batch):
            """Estimate GPU memory needed for processing n_batch atoms simultaneously.
            
            For batch processing, peak memory includes:
            - Real space intermediates: delta_real, r2_real, mask, erfc_term
              Size: n_grid * n_batch * n_shifts * (3 + 1 + 1 + 1) * dtype_size
            - Reciprocal space intermediates: delta_recp, kr, cos_kr  
              Size: n_grid * n_batch * (3 + n_k + n_k) * dtype_size
            - Result buffer: n_grid * n_batch * dtype_size
            - Atoms buffer: n_batch * 3 * dtype_size
            """
            # Real space (4 arrays: delta components implicit in r2 calc)
            real_space = n_grid * n_batch * n_shifts * 4 * dtype_size
            
            # Reciprocal space (3 arrays: delta, kr, cos_kr)
            recp_space = n_grid * n_batch * (3 + 2 * n_k) * dtype_size
            
            # Output and input buffers
            buffers = n_grid * n_batch * dtype_size + n_batch * 3 * dtype_size
            
            return real_space + recp_space + buffers
        
        def estimate_single_atom_memory():
            """Estimate memory for current single-atom-at-a-time approach."""
            # Current implementation uses these intermediates per atom:
            # Real: delta_real [n_grid, n_shifts, 3], r2_real [n_grid, n_shifts], 
            #       mask [n_grid, n_shifts], erfc_term [n_grid, n_shifts]
            # Recp: delta_recp [n_grid, 3], kr [n_grid, n_k], cos_kr [n_grid, n_k]
            real = n_grid * n_shifts * (3 + 1 + 1 + 1) * dtype_size
            recp = n_grid * (3 + n_k + n_k) * dtype_size
            return real + recp + n_grid * dtype_size  # Add result column
        
        # Check available GPU memory and determine batch size
        def get_gpu_memory_info(device_id):
            with cp.cuda.Device(device_id):
                mem_info = cp.cuda.runtime.memGetInfo()
                free_bytes = mem_info[0]
                total_bytes = mem_info[1]
                # Reserve 20% for overhead
                usable_bytes = int(free_bytes * 0.8)
                return usable_bytes, total_bytes
        
        # Calculate optimal batch size per GPU
        gpu_memory_info = []
        for device_id in devices:
            usable, total = get_gpu_memory_info(device_id)
            gpu_memory_info.append((device_id, usable, total))
            
        min_usable = min(info[1] for info in gpu_memory_info)
        
        # Memory needed for constants (grid_pos, shifts, kvecs, kcoefs)
        constant_memory = bytes_per_grid + bytes_per_shifts + bytes_per_kvecs + bytes_per_kcoefs
        
        # Memory available for atom batches
        available_for_atoms = min_usable - constant_memory
        
        # Calculate optimal batch size using binary search
        # Find the largest batch size that fits in available memory
        def find_optimal_batch_size():
            """Find maximum batch size that fits in available memory."""
            if available_for_atoms <= 0:
                return 1  # Fallback to single atom
            
            # Binary search for optimal batch size
            low, high = 1, max(1, n_atoms // n_gpus)
            best_batch = 1
            
            while low <= high:
                mid = (low + high) // 2
                mem_needed = estimate_batch_memory(mid)
                
                if mem_needed <= available_for_atoms * 0.8:  # 80% safety margin
                    best_batch = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            return best_batch
        
        # Current single-atom memory usage
        bytes_per_single_atom = estimate_single_atom_memory()
        
        if available_for_atoms < bytes_per_single_atom * 2:
            # Fall back to CPU if GPU memory insufficient
            print(f"    WARNING: GPU memory insufficient ({min_usable/1e9:.2f} GB available)")
            print(f"    Falling back to sequential CPU computation...")
            return self._compute_sequential(grid_pos, atom_positions, use_jit=True, progress=progress)
        
        # Calculate batch size based on memory availability
        atoms_per_batch = find_optimal_batch_size()
        
        # Determine batch strategy for reporting
        batch_memory = estimate_batch_memory(atoms_per_batch)
        if atoms_per_batch >= n_atoms // n_gpus:
            batch_strategy = "single-batch (memory sufficient)"
        elif batch_memory >= available_for_atoms * 0.5:
            batch_strategy = "large-batch"
        else:
            batch_strategy = f"batch-size-{atoms_per_batch} (memory limited)"
        
        if progress:
            batch_mem_gb = estimate_batch_memory(atoms_per_batch) / 1e9
            const_mem_gb = constant_memory / 1e9
            avail_gb = available_for_atoms / 1e9
            print(f"    GPU memory: {avail_gb:.2f} GB available (constants: {const_mem_gb:.2f} GB)")
            print(f"    Batch size: {atoms_per_batch} atoms ({batch_mem_gb:.2f} GB per batch)")
            print(f"    Strategy: {batch_strategy}")
            print(f"    Estimated transfers: {max(1, (n_atoms // n_gpus + atoms_per_batch - 1) // atoms_per_batch)} per GPU")
        
        # Use pinned memory for faster CPU-GPU transfers
        # Allocate pinned memory and create numpy array view
        grid_pos_pinned = cp.cuda.alloc_pinned_memory(grid_pos.nbytes)
        grid_pos_np = np.frombuffer(grid_pos_pinned, dtype=grid_pos.dtype, count=grid_pos.size).reshape(grid_pos.shape)
        np.copyto(grid_pos_np, grid_pos)
        
        # Result buffer (pinned memory) - allocate in chunks
        result_dtype = np.float32 if use_fp32 else np.float64
        phi = np.zeros((n_grid, n_atoms), dtype=np.float64)  # Always return fp64
        
        # Setup GPU data structures
        gpu_data = []
        for device_id in devices:
            with cp.cuda.Device(device_id):
                stream = cp.cuda.Stream(non_blocking=True)
                
                # Allocate constants
                # Convert to target dtype during transfer
                grid_pos_gpu = cp.asarray(grid_pos_np, dtype=cp_dtype)
                
                kvecs_gpu = cp.asarray(self.kvecs, dtype=cp_dtype)
                kcoefs_gpu = cp.asarray(self.kcoefs, dtype=cp_dtype)
                shifts_gpu = cp.asarray(self.shifts, dtype=cp_dtype)
                
                # Pre-allocate working buffers (reused across batches)
                max_batch_atoms = min(atoms_per_batch, n_atoms)
                result_buffer = cp.empty((n_grid, max_batch_atoms), dtype=cp_dtype)
                atoms_buffer = cp.empty((max_batch_atoms, 3), dtype=cp_dtype)
                
                gpu_data.append({
                    'device_id': device_id,
                    'stream': stream,
                    'grid_pos': grid_pos_gpu,
                    'kvecs': kvecs_gpu,
                    'kcoefs': kcoefs_gpu,
                    'shifts': shifts_gpu,
                    'result_buffer': result_buffer,
                    'atoms_buffer': atoms_buffer,
                    'max_batch_atoms': max_batch_atoms,
                })
        
        if progress:
            print(f"    GPU: Computing...")
            start_time = time.time()
            last_update = start_time
        
        # Use a list to track progress (avoid nonlocal in nested function)
        progress_tracker = [0]  # progress_tracker[0] = completed_atoms
        
        def compute_batch(gpu_idx, atom_start, atom_end):
            """Compute a batch of atoms on a specific GPU with vectorized shifts and k-vectors."""
            data = gpu_data[gpu_idx]
            device_id = data['device_id']
            stream = data['stream']
            
            n_batch_atoms = atom_end - atom_start
            
            with cp.cuda.Device(device_id):
                # Transfer atom positions for this batch
                atom_batch = atom_positions[atom_start:atom_end]
                # Convert to target dtype during transfer to avoid type mismatch
                atom_batch_cp = cp.asarray(atom_batch, dtype=cp_dtype)
                data['atoms_buffer'][:n_batch_atoms] = atom_batch_cp
                
                # Pre-compute constants as Python floats (not CuPy scalars)
                sqrt_alpha_val = float(self.sqrt_alpha)
                R2_val = float(self.R_cutoff ** 2)
                
                # Find zero shift index for self-interaction exclusion
                # Check on CPU to avoid GPU-CPU sync in loop
                shifts_cpu = cp.asnumpy(data['shifts'])
                zero_shift_idx = -1
                for idx, shift in enumerate(shifts_cpu):
                    if shift[0] == 0.0 and shift[1] == 0.0 and shift[2] == 0.0:
                        zero_shift_idx = idx
                        break
                
                # Compute all atoms in batch
                for local_i in range(n_batch_atoms):
                    atom_pos_gpu = data['atoms_buffer'][local_i]
                    
                    # === Real Space Contribution (Vectorized) ===
                    # Compute distances to all image positions at once
                    # [n_grid, 1, 3] - [1, n_shifts, 3] -> [n_grid, n_shifts, 3]
                    img_positions = atom_pos_gpu + data['shifts']  # [n_shifts, 3]
                    delta_real = data['grid_pos'][:, None, :] - img_positions[None, :, :]  # [n_grid, n_shifts, 3]
                    r2_real = cp.sum(delta_real ** 2, axis=2)  # [n_grid, n_shifts]
                    
                    # Apply cutoff mask
                    mask = r2_real <= R2_val
                    
                    # Exclude self-interaction for zero shift
                    if zero_shift_idx >= 0:
                        # Create a copy of mask and exclude r2 < 1e-20 for zero shift
                        mask = mask.copy()
                        mask[:, zero_shift_idx] = mask[:, zero_shift_idx] & (r2_real[:, zero_shift_idx] > 1e-20)
                    
                    # Compute erfc/r for all valid points (where mask is True)
                    # Use cp.where to avoid computing sqrt for all points
                    r_real = cp.sqrt(r2_real)
                    # Safe division: where mask is False, result is 0
                    erfc_term = cp.where(mask, cupy_erfc(sqrt_alpha_val * r_real) / r_real, 0.0)
                    
                    # Sum over all shifts
                    phi_real = cp.sum(erfc_term, axis=1)  # [n_grid]
                    
                    # === Reciprocal Space Contribution (Vectorized) ===
                    # delta: [n_grid, 3]
                    delta_recp = data['grid_pos'] - atom_pos_gpu  # [n_grid, 3]
                    
                    # Compute k·r for all k-vectors using matrix multiplication
                    # [n_grid, 3] @ [3, n_k] -> [n_grid, n_k]
                    kr = delta_recp @ data['kvecs'].T
                    cos_kr = cp.cos(kr)  # [n_grid, n_k]
                    
                    # Weighted sum over k-vectors
                    # [n_grid, n_k] @ [n_k] -> [n_grid]
                    phi_recp = cos_kr @ data['kcoefs']
                    
                    # Combine and store
                    data['result_buffer'][:, local_i] = phi_real + phi_recp
                
                # Transfer results back
                stream.synchronize()
                result_gpu = data['result_buffer'][:, :n_batch_atoms]
                result_slice = phi[:, atom_start:atom_end]
                result_slice[:] = cp.asnumpy(result_gpu)
                
                return n_batch_atoms
        
        # Process all batches
        if n_gpus > 1:
            # Multi-GPU: distribute batches across GPUs
            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = []
                atom_idx = 0
                
                while atom_idx < n_atoms:
                    for gpu_idx in range(n_gpus):
                        if atom_idx >= n_atoms:
                            break
                        
                        batch_start = atom_idx
                        batch_end = min(batch_start + atoms_per_batch, n_atoms)
                        
                        future = executor.submit(compute_batch, gpu_idx, batch_start, batch_end)
                        batch_size = batch_end - batch_start
                        futures.append((future, batch_size))
                        
                        atom_idx = batch_end
                    
                    # Wait for current wave to complete
                    for future, batch_size in futures:
                        completed = future.result()
                        progress_tracker[0] += completed
                    futures = []
                    
                    # Progress update
                    if progress:
                        current_time = time.time()
                        if current_time - last_update > 1.0:  # Update every second
                            completed = progress_tracker[0]
                            pct = 100 * completed / n_atoms
                            elapsed = current_time - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            print(f"    Progress: {completed}/{n_atoms} ({pct:.1f}%) [{elapsed:.1f}s, {rate:.1f} atoms/s]")
                            last_update = current_time
        else:
            # Single GPU: process all batches sequentially
            atom_idx = 0
            while atom_idx < n_atoms:
                batch_end = min(atom_idx + atoms_per_batch, n_atoms)
                completed = compute_batch(0, atom_idx, batch_end)
                progress_tracker[0] += completed
                
                if progress:
                    current_time = time.time()
                    if current_time - last_update > 1.0:
                        completed = progress_tracker[0]
                        pct = 100 * completed / n_atoms
                        elapsed = current_time - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"    Progress: {completed}/{n_atoms} ({pct:.1f}%) [{elapsed:.1f}s, {rate:.1f} atoms/s]")
                        last_update = current_time
                
                atom_idx = batch_end
        
        if progress:
            elapsed = time.time() - start_time
            print(f"    Complete: {n_atoms}/{n_atoms} (100%) [{elapsed:.2f}s]")
        
        # Clean up GPU memory
        for data in gpu_data:
            device_id = data['device_id']
            with cp.cuda.Device(device_id):
                # Delete all GPU arrays
                for key in ['grid_pos', 'kvecs', 'kcoefs', 'shifts', 'result_buffer', 'atoms_buffer']:
                    if key in data:
                        del data[key]
                # Free memory pool
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        
        # Clean up pinned memory
        del grid_pos_pinned
        
        return phi
    
    def _compute_gpu_torch(self, grid_pos, atom_positions, devices, use_fp32, progress):
        """Compute using PyTorch with multi-GPU support."""
        import torch
        from concurrent.futures import ThreadPoolExecutor
        
        n_grid = len(grid_pos)
        n_atoms = len(atom_positions)
        n_gpus = len(devices)
        
        # Choose dtype
        torch_dtype = torch.float32 if use_fp32 else torch.float64
        np_dtype = np.float32 if use_fp32 else np.float64
        
        # Transfer common data to all GPUs
        grid_pos_gpu_list = []
        kvecs_gpu_list = []
        kcoefs_gpu_list = []
        shifts_gpu_list = []
        
        for device_id in devices:
            device = torch.device(f'cuda:{device_id}')
            grid_pos_gpu_list.append(torch.tensor(grid_pos, dtype=torch_dtype, device=device))
            kvecs_gpu_list.append(torch.tensor(self.kvecs, dtype=torch_dtype, device=device))
            kcoefs_gpu_list.append(torch.tensor(self.kcoefs, dtype=torch_dtype, device=device))
            shifts_gpu_list.append(torch.tensor(self.shifts, dtype=torch_dtype, device=device))
        
        # Result array
        phi = np.zeros((n_grid, n_atoms), dtype=np_dtype)
        
        # Divide atoms among GPUs
        atoms_per_gpu = (n_atoms + n_gpus - 1) // n_gpus
        
        def compute_atom_range(gpu_idx, atom_start, atom_end):
            """Compute a range of atoms on a specific GPU."""
            device_id = devices[gpu_idx]
            device = torch.device(f'cuda:{device_id}')
            
            grid_pos_gpu = grid_pos_gpu_list[gpu_idx]
            kvecs_gpu = kvecs_gpu_list[gpu_idx]
            kcoefs_gpu = kcoefs_gpu_list[gpu_idx]
            shifts_gpu = shifts_gpu_list[gpu_idx]
            
            sqrt_alpha_val = torch.tensor(self.sqrt_alpha, dtype=torch_dtype, device=device)
            R2 = torch.tensor(self.R_cutoff ** 2, dtype=torch_dtype, device=device)
            
            local_phi = torch.zeros((n_grid, atom_end - atom_start), dtype=torch_dtype, device=device)
            
            for local_i, atom_idx in enumerate(range(atom_start, atom_end)):
                atom_pos = atom_positions[atom_idx]
                atom_pos_gpu = torch.tensor(atom_pos, dtype=torch_dtype, device=device)
                
                # Real space contribution
                phi_real = torch.zeros(n_grid, dtype=torch_dtype, device=device)
                
                for s in range(shifts_gpu.shape[0]):
                    shift = shifts_gpu[s]
                    img_pos = atom_pos_gpu + shift
                    delta = grid_pos_gpu - img_pos
                    r2 = torch.sum(delta ** 2, dim=1)
                    
                    mask = r2 <= R2
                    if s == 0:
                        mask = mask & (r2 > 1e-20)
                    
                    if torch.any(mask):
                        r = torch.sqrt(r2[mask])
                        phi_real[mask] += torch.erfc(sqrt_alpha_val * r) / r
                
                # Reciprocal space contribution
                delta = grid_pos_gpu - atom_pos_gpu
                phi_recp = torch.zeros(n_grid, dtype=torch_dtype, device=device)
                
                for j in range(len(kvecs_gpu)):
                    kvec = kvecs_gpu[j]
                    kcoef = kcoefs_gpu[j]
                    kr = torch.sum(delta * kvec, dim=1)
                    phi_recp += kcoef * torch.cos(kr)
                
                local_phi[:, local_i] = phi_real + phi_recp
            
            return global_i, local_phi.cpu().numpy()
        
        # Parallel computation across GPUs
        if n_gpus > 1:
            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = []
                for gpu_idx in range(n_gpus):
                    atom_start = gpu_idx * atoms_per_gpu
                    atom_end = min(atom_start + atoms_per_gpu, n_atoms)
                    if atom_start < atom_end:
                        futures.append(executor.submit(compute_atom_range, gpu_idx, atom_start, atom_end))
                
                # Collect results
                for i, future in enumerate(futures):
                    last_atom_idx, local_phi = future.result()
                    atom_start = i * atoms_per_gpu
                    atom_end = min(atom_start + atoms_per_gpu, n_atoms)
                    phi[:, atom_start:atom_end] = local_phi
                    
                    if progress:
                        pct = 100 * atom_end / n_atoms
                        print(f"    GPU batch complete: {atom_end}/{n_atoms} ({pct:.0f}%)")
        else:
            # Single GPU
            _, local_phi = compute_atom_range(0, 0, n_atoms)
            phi = local_phi
        
        # Clean up GPU memory
        if n_gpus > 1:
            for gpu_idx in range(n_gpus):
                device_id = devices[gpu_idx]
                device = torch.device(f'cuda:{device_id}')
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
        
        return phi.astype(np.float64)  # Return in fp64 for compatibility
    
    def _compute_nogil_threads(self, grid_pos, atom_positions, n_cores=None, progress=True):
        """Compute using noGIL threading - shared memory, no serialization overhead."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from functools import partial
        
        n_grid = len(grid_pos)
        n_atoms = len(atom_positions)
        
        if n_cores is None:
            n_cores = _get_optimal_workers()
        
        phi = np.zeros((n_grid, n_atoms), dtype=np.float64)
        grid_pos_arr = np.asarray(grid_pos, dtype=np.float64)
        atom_positions_arr = np.asarray(atom_positions, dtype=np.float64)
        
        if progress:
            print(f"  Computing with noGIL threads ({n_cores} workers)...")
        
        # Prepare worker function with bound arguments
        worker_func = partial(
            _compute_ewald_atom_worker,
            grid_pos=grid_pos_arr,
            kvecs=self.kvecs,
            kcoefs=self.kcoefs,
            shifts=self.shifts,
            R_cutoff=self.R_cutoff,
            sqrt_alpha=self.sqrt_alpha,
            fit_flag=self.fit_flag,
            block_k=self.block_k,
            use_jit=self.use_jit,
        )
        
        # Create job list
        jobs = [(idx, atom_positions_arr[idx]) for idx in range(n_atoms)]
        
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = {executor.submit(worker_func, idx, pos): idx for idx, pos in jobs}
            
            done_count = 0
            for future in as_completed(futures):
                atom_idx, col = future.result()
                phi[:, atom_idx] = col
                done_count += 1
                if progress and done_count % max(1, n_atoms // 10) == 0:
                    pct = 100 * done_count / n_atoms
                    print(f"    Progress: {done_count}/{n_atoms} ({pct:.0f}%)", flush=True)
        
        return phi
    
    def _compute_numba_parallel(self, grid_pos, atom_positions, progress=True):
        """Compute using Numba parallel (prange) - best for large grids."""
        n_grid = len(grid_pos)
        n_atoms = len(atom_positions)
        
        phi = np.zeros((n_grid, n_atoms), dtype=np.float64)
        grid_pos_arr = np.asarray(grid_pos, dtype=np.float64)
        
        if progress:
            print(f"  Computing {n_atoms} atoms with Numba parallel...")
        
        for i, atom_pos in enumerate(atom_positions):
            delta = grid_pos_arr - atom_pos
            
            # Use parallel JIT functions
            phi_recp = _reciprocal_sum_blocked_jit_parallel(
                delta, self.kvecs, self.kcoefs, self.block_k
            )
            phi_real = _real_space_sum_jit_parallel(
                grid_pos_arr, atom_pos, self.shifts, 
                self.R_cutoff, self.sqrt_alpha
            )
            
            phi[:, i] = phi_recp + phi_real
            
            if progress and (i + 1) % max(1, n_atoms // 10) == 0:
                pct = 100 * (i + 1) / n_atoms
                print(f"    Progress: {i+1}/{n_atoms} ({pct:.0f}%)")
        
        return phi
    
    def _compute_sequential(self, grid_pos, atom_positions, use_jit=True, progress=True):
        """Compute sequentially - best for small systems."""
        n_grid = len(grid_pos)
        n_atoms = len(atom_positions)
        
        phi = np.zeros((n_grid, n_atoms), dtype=np.float64)
        grid_pos_arr = np.asarray(grid_pos, dtype=np.float64)
        atom_positions_arr = np.asarray(atom_positions, dtype=np.float64)
        
        if progress:
            print(f"  Computing {n_atoms} atoms sequentially...")
        
        for i in range(n_atoms):
            atom_pos = atom_positions_arr[i]
            delta = grid_pos_arr - atom_pos
            
            if use_jit and _JIT_OK:
                phi_recp = _reciprocal_sum_blocked_jit(
                    delta, self.kvecs, self.kcoefs, self.block_k
                )
                phi_real = _real_space_sum_jit(
                    grid_pos_arr, atom_pos, self.shifts,
                    self.R_cutoff, self.sqrt_alpha
                )
            else:
                phi_recp = _reciprocal_sum_blocked(
                    delta, self.kvecs, self.kcoefs, self.block_k, use_jit=False
                )
                phi_real = _real_space_sum(
                    grid_pos_arr, atom_pos, self.shifts,
                    self.R_cutoff, self.sqrt_alpha, use_jit=False
                )
            
            phi[:, i] = phi_recp + phi_real
            
            if progress and (i + 1) % max(1, n_atoms // 10) == 0:
                pct = 100 * (i + 1) / n_atoms
                print(f"    Progress: {i+1}/{n_atoms} ({pct:.0f}%)")
        
        return phi
    
    def _compute_multiprocess(self, grid_pos, atom_positions, n_cores=None, progress=True,
                              adaptive_block=True, min_block_k=8, max_retries=10):
        """Compute using multiprocessing - best for medium systems."""
        use_threads = _NOGIL_ENABLED
        
        # Check memory
        if psutil is not None:
            available_ram = psutil.virtual_memory().available
        else:
            available_ram = 4 * 1024 * 1024 * 1024
        
        n_atoms = len(atom_positions)
        n_grid = len(grid_pos)
        
        phi_bytes = n_grid * n_atoms * 8
        if phi_bytes > available_ram * 0.9:
            raise MemoryError(
                f"Not enough RAM for result matrix. Need {phi_bytes/1e9:.2f} GB"
            )
        
        remaining_ram = available_ram - phi_bytes
        worker_base_overhead = 256 * 1024 * 1024 if use_threads else 512 * 1024 * 1024
        
        block_k = int(self.block_k)
        last_exc = None
        
        for _attempt in range(max_retries if adaptive_block else 1):
            try:
                # Determine batch size
                temp_bytes_per_atom = n_grid * block_k * 8
                safe_worker_ram = remaining_ram * 0.8
                estimated_mem_per_worker = worker_base_overhead + temp_bytes_per_atom
                max_workers_by_mem = int(safe_worker_ram // max(1, estimated_mem_per_worker))
                effective_cores = min(n_cores, max(1, max_workers_by_mem))
                
                # Batch sizing
                MIN_BATCH_SIZE = 1
                if effective_cores > 1:
                    max_batch_size = max(MIN_BATCH_SIZE, n_atoms // effective_cores)
                    target_batches = effective_cores * (2 if use_threads else 3)
                    batch_size_from_target = max(MIN_BATCH_SIZE, n_atoms // max(1, target_batches))
                    batch_size = min(max_batch_size, batch_size_from_target)
                    
                    max_chunk_bytes = 1024 * 1024 * 1024 if use_threads else 512 * 1024 * 1024
                    bytes_per_atom_res = n_grid * 8
                    max_batch_by_ipc = max(MIN_BATCH_SIZE, int(max_chunk_bytes // max(1, bytes_per_atom_res)))
                    batch_size = min(batch_size, max_batch_by_ipc)
                    batch_size = min(batch_size, n_atoms)
                else:
                    batch_size = max(MIN_BATCH_SIZE, n_atoms)
                
                # Prepare data
                grid_pos_arr = np.asarray(grid_pos, dtype=np.float64)
                atom_positions_arr = np.asarray(atom_positions, dtype=np.float64)
                
                phi = np.zeros((n_grid, n_atoms), dtype=np.float64)
                
                jobs = [(idx, atom_positions_arr[idx]) for idx in range(n_atoms)]
                total_jobs = len(jobs)
                
                if effective_cores <= 1 or total_jobs <= 1:
                    # Fall back to sequential
                    return self._compute_sequential(grid_pos, atom_positions, use_jit=True, progress=progress)
                
                # Execute with appropriate backend
                if use_threads and _NOGIL_ENABLED:
                    worker_func = partial(
                        _compute_ewald_atom_worker,
                        grid_pos=grid_pos_arr,
                        kvecs=self.kvecs,
                        kcoefs=self.kcoefs,
                        shifts=self.shifts,
                        R_cutoff=self.R_cutoff,
                        sqrt_alpha=self.sqrt_alpha,
                        fit_flag=self.fit_flag,
                        block_k=block_k,
                        use_jit=self.use_jit,
                    )
                    
                    with ThreadPoolExecutor(max_workers=effective_cores) as executor:
                        futures = {executor.submit(worker_func, idx, pos): idx for idx, pos in jobs}
                        
                        done_count = 0
                        for future in as_completed(futures):
                            atom_idx, col = future.result()
                            phi[:, atom_idx] = col
                            done_count += 1
                            if progress and done_count % max(1, total_jobs // 10) == 0:
                                pct = 100 * done_count / total_jobs
                                print(f"    Batch Progress: {done_count}/{total_jobs} ({pct:.0f}%)", flush=True)
                else:
                    # Process-based
                    mp_ctx = _pick_parallel_backend(prefer_threads=False)[2]
                    
                    batch_jobs = []
                    for start in range(0, n_atoms, batch_size):
                        end = min(start + batch_size, n_atoms)
                        batch_jobs.append((start, jobs[start:end]))
                    
                    total_batches = len(batch_jobs)
                    if total_batches <= effective_cores:
                        chunksize = 1
                    else:
                        chunksize = max(1, total_batches // effective_cores // 2)
                    
                    if progress:
                        print(f"    Parallel config: {effective_cores} processes, {total_batches} batches, chunksize={chunksize}")
                    
                    with mp_ctx.Pool(processes=effective_cores) as pool:
                        it = pool.imap_unordered(
                            _compute_ewald_batch_wrapper,
                            [(*batch, grid_pos_arr, self.kvecs, self.kcoefs, self.shifts,
                              self.R_cutoff, self.sqrt_alpha, self.fit_flag, block_k, self.use_jit)
                             for batch in batch_jobs],
                            chunksize=chunksize
                        )
                        
                        done_count = 0
                        for batch_result in it:
                            for atom_idx, col in batch_result:
                                phi[:, atom_idx] = col
                            done_count += 1
                            if progress and done_count % max(1, total_batches // 10) == 0:
                                pct = 100 * done_count / total_batches
                                print(f"    Batch Progress: {done_count}/{total_batches} ({pct:.0f}%)", flush=True)
                
                return phi
                
            except MemoryError as e:
                last_exc = e
                if not adaptive_block or block_k <= min_block_k:
                    raise
                new_block = max(min_block_k, block_k // 2)
                if progress:
                    print(f"MemoryError: reducing block_k {block_k} -> {new_block} and retry ...")
                block_k = new_block
                
            except Exception as e:
                last_exc = e
                if self.use_jit:
                    if progress:
                        print(f"Warning: execution failed ({e}); disabling JIT and retry ...")
                    self.use_jit = False
                    continue
                
                # Fallback to sequential
                return self._compute_sequential(grid_pos, atom_positions, use_jit=False, progress=progress)
        
        raise last_exc if last_exc else RuntimeError("Unknown error in compute_batch_parallel")

class ChargeCalculator:
    """Main calculator with streaming support for large systems."""

    def __init__(self, cube_file, fit_flag=1, vdw_factor=1.0, vdw_max=1000.0,
                 R_cutoff=20.0, q_tot=0.0, symm_file=None, resp_file=None,
                 qeq_file=None, n_cores=None, block_k=64, use_jit=True):
        self.cube_file = cube_file
        self.fit_flag = int(fit_flag)
        self.vdw_factor = float(vdw_factor)
        self.vdw_max = float(vdw_max)
        self.R_cutoff = float(R_cutoff)
        self.q_tot = float(q_tot)
        self.symm_file = symm_file
        self.resp_file = resp_file
        self.qeq_file = qeq_file

        self.cube = self._read_cube(cube_file)
        self.n_atoms = self.cube["n_atoms"]

        n_grid_total = int(np.prod(self.cube["n_grid"]))
        self.n_cores = determine_optimal_cores(
            n_atoms=self.n_atoms,
            n_grid=n_grid_total,
            n_cores_requested=n_cores
        )

        self.box_vectors = np.array([
            self.cube["n_grid"][i] * self.cube["axis_vector"][i]
            for i in range(3)
        ], dtype=np.float64)
        self.volume = float(np.dot(self.box_vectors[0], np.cross(self.box_vectors[1], self.box_vectors[2])))

        self.ewald = EwaldCalculator(
            self.box_vectors, self.volume, self.R_cutoff, self.fit_flag, self.n_cores,
            block_k=block_k, use_jit=use_jit
        )

        self._neighbor_shifts_27 = _make_neighbor_shifts_27(self.box_vectors)
        
        t0 = time.time()
        self.grid_pos, self.V_pot = self._filter_grid()
        print(f"  vdW filter completed in {time.time() - t0:.2f} seconds")

        self.symm_data = self._read_symmetry(symm_file) if symm_file else None
        self.resp_data = self._read_resp_params(resp_file) if resp_file else None
        self.qeq_data = self._read_qeq_params(qeq_file) if qeq_file else None

        # Removed self._phi to enforce streaming logic
        
    # -------------------- IO (Unchanged) --------------------

    def _read_cube(self, filename):
        with open(filename, 'r') as f:
            f.readline()
            f.readline()

            parts = f.readline().split()
            n_atoms = int(parts[0])

            n_grid = np.zeros(3, dtype=int)
            axis_vector = np.zeros((3, 3))
            for i in range(3):
                parts = f.readline().split()
                n_grid[i] = int(parts[0])
                axis_vector[i] = [float(x) for x in parts[1:4]]

            atom_index = np.zeros(n_atoms, dtype=int)
            atom_pos = np.zeros((n_atoms, 3))
            for i in range(n_atoms):
                parts = f.readline().split()
                atom_index[i] = int(parts[0])
                atom_pos[i] = [float(x) for x in parts[2:5]]

            data = [float(x) for line in f for x in line.split()]
            V_pot = np.zeros((n_grid[0] * n_grid[1], n_grid[2]))
            idx = 0
            for i in range(n_grid[0]):
                for j in range(n_grid[1]):
                    i_2d = i + j * n_grid[0]
                    for k in range(n_grid[2]):
                        V_pot[i_2d, k] = data[idx]
                        idx += 1

        return {
            "n_atoms": n_atoms,
            "n_grid": n_grid,
            "atom_index": atom_index,
            "atom_pos": atom_pos,
            "axis_vector": axis_vector,
            "V_pot": V_pot
        }

    def _parse_atom_ranges(self, range_str):
        """Parse atom range string like '1-5,9,10,15-18' to list of indices.
        
        Args:
            range_str: String with atom indices/ranges (1-based, comma-separated)
            
        Returns:
            List of 0-based atom indices
        """
        indices = []
        for part in range_str.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                start, end = part.split('-', 1)
                start, end = int(start.strip()), int(end.strip())
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        # Convert to 0-based and remove duplicates while preserving order
        seen = set()
        result = []
        for idx in indices:
            idx_0based = idx - 1
            if idx_0based not in seen:
                seen.add(idx_0based)
                result.append(idx_0based)
        return result
    
    def _read_symmetry(self, filename):
        """Read symmetry constraints from file.
        
        Format (NEW - simplified):
            # Comments start with #
            1-5,9,10        # Equivalent atoms group 1
            15-18           # Equivalent atoms group 2
            20,21,22        # Equivalent atoms group 3
            
        Each line defines one group of equivalent atoms.
        Supports ranges (1-5), single indices (9), comma-separated lists.
        First atom in each group is the base, others are linked.
        """
        try:
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            trees = []
            for line_num, line in enumerate(lines, 1):
                # Remove inline comments
                if '#' in line:
                    line = line.split('#')[0].strip()
                if not line:
                    continue
                    
                atom_indices = self._parse_atom_ranges(line)
                if len(atom_indices) < 1:
                    print(f"Warning: Line {line_num} has no valid atoms, skipping")
                    continue
                    
                trees.append({
                    'n_equiv': len(atom_indices),
                    'base': atom_indices[0],
                    'linked': atom_indices[1:] if len(atom_indices) > 1 else []
                })
            
            n_trees = len(trees)
            if n_trees > 0:
                print(f"  Symmetry constraints loaded: {n_trees} groups")
                for i, tree in enumerate(trees):
                    base_idx = tree['base'] + 1  # Convert back to 1-based for display
                    linked_str = ','.join(str(x + 1) for x in tree['linked']) if tree['linked'] else 'none'
                    print(f"    Group {i+1}: base={base_idx}, linked=[{linked_str}]")
            return {'trees': trees, 'n_independent': n_trees}
        except Exception as e:
            print(f"Warning: Error reading symmetry file: {e}")
            return None

    def _read_resp_params(self, filename):
        resp_params = {}
        try:
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        elem = parts[0]
                        resp_params[elem] = {'strength': float(parts[1]), 'target': float(parts[2])}
            print(f"  RESP parameters loaded from {filename}")
            return resp_params
        except Exception:
            return {}

    def _read_qeq_params(self, filename):
        qeq_params = {}
        try:
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        elem = parts[0]
                        qeq_params[elem] = {'electroneg': float(parts[1]), 'hardness': float(parts[2])}
            print(f"  QEq parameters loaded from {filename}")
            return qeq_params
        except Exception:
            return {}

    # -------------------- Grid Filtering --------------------

    def _filter_grid_gpu_single(self, eff_atoms_pos, eff_radii_min, eff_radii_max,
                                   base_xy, axis, nz, nx, ny, device_id=0):
        """
        Single GPU grid filtering using specified device with pinned memory optimization.
        
        Args:
            device_id: GPU device ID to use
        """
        import cupy as cp
        
        # Set the device for this operation
        with cp.cuda.Device(device_id):
            n_atoms = len(eff_atoms_pos)
            n_xy = nx * ny
            n_total = n_xy * nz
            
            # Check memory on THIS device
            free_mem = cp.cuda.Device(device_id).mem_info[0]
            bytes_per_point = 24 + 8 * n_atoms * 2
            bytes_needed = n_total * bytes_per_point
            
            # Use pinned memory for faster CPU->GPU transfers
            # Allocate pinned memory and copy data
            atoms_pos_pinned = cp.cuda.alloc_pinned_memory(eff_atoms_pos.nbytes)
            atoms_pos_np = np.frombuffer(atoms_pos_pinned, dtype=eff_atoms_pos.dtype, 
                                         count=eff_atoms_pos.size).reshape(eff_atoms_pos.shape)
            np.copyto(atoms_pos_np, eff_atoms_pos)
            
            radii_min_pinned = cp.cuda.alloc_pinned_memory(eff_radii_min.nbytes)
            radii_min_np = np.frombuffer(radii_min_pinned, dtype=eff_radii_min.dtype,
                                         count=eff_radii_min.size)
            np.copyto(radii_min_np, eff_radii_min)
            
            radii_max_pinned = cp.cuda.alloc_pinned_memory(eff_radii_max.nbytes)
            radii_max_np = np.frombuffer(radii_max_pinned, dtype=eff_radii_max.dtype,
                                         count=eff_radii_max.size)
            np.copyto(radii_max_np, eff_radii_max)
            
            base_xy_pinned = cp.cuda.alloc_pinned_memory(base_xy.nbytes)
            base_xy_np = np.frombuffer(base_xy_pinned, dtype=base_xy.dtype,
                                       count=base_xy.size).reshape(base_xy.shape)
            np.copyto(base_xy_np, base_xy)
            
            z_vec_pinned = cp.cuda.alloc_pinned_memory(axis[2].nbytes)
            z_vec_np = np.frombuffer(z_vec_pinned, dtype=axis[2].dtype,
                                     count=axis[2].size)
            np.copyto(z_vec_np, axis[2])
            
            # Transfer data to GPU from pinned memory
            atoms_pos_gpu = cp.asarray(atoms_pos_np, dtype=cp.float64)
            radii_min_gpu = cp.asarray(radii_min_np, dtype=cp.float64)
            radii_max_gpu = cp.asarray(radii_max_np, dtype=cp.float64)
            base_xy_gpu = cp.asarray(base_xy_np, dtype=cp.float64)
            z_vec = cp.asarray(z_vec_np, dtype=cp.float64)
            
            # Pre-compute atoms_norm
            atoms_norm = cp.sum(atoms_pos_gpu ** 2, axis=1)
            
            # Use 40% of memory for single-pass processing
            if bytes_needed <= free_mem * 0.4:
                # Single-pass processing
                z_range = cp.arange(nz, dtype=cp.float64)
                z_grid = z_range[:, None] * z_vec[None, :]
                grid_pos = base_xy_gpu[None, :, :] + z_grid[:, None, :]
                grid_pos = grid_pos.reshape(-1, 3)
                
                grid_norm = cp.sum(grid_pos ** 2, axis=1, keepdims=True)
                dot_product = grid_pos @ atoms_pos_gpu.T
                dists_sq = grid_norm + atoms_norm - 2 * dot_product
                dists = cp.sqrt(cp.maximum(dists_sq, 0))
                
                valid_mask = cp.all(dists > radii_min_gpu[None, :], axis=1)
                near_mask = cp.any(dists <= radii_max_gpu[None, :], axis=1)
                
                valid_mask_cpu = valid_mask.get()
                near_mask_cpu = near_mask.get()
                
                # Clean up GPU memory before returning
                del (grid_pos, grid_norm, dot_product, dists_sq, dists, 
                     valid_mask, near_mask, z_range, z_grid)
                del atoms_pos_gpu, radii_min_gpu, radii_max_gpu
                del base_xy_gpu, z_vec, atoms_norm
                cp.get_default_memory_pool().free_all_blocks()
                
                # Clean up pinned memory
                del atoms_pos_pinned, radii_min_pinned, radii_max_pinned
                del base_xy_pinned, z_vec_pinned
                
                return valid_mask_cpu, near_mask_cpu, True
            else:
                # Need chunked processing on this device
                chunk_z = max(1, int((free_mem * 0.5 / bytes_per_point) / n_xy))
                chunk_z = min(chunk_z, nz)
                
                valid_mask_all = np.empty(n_total, dtype=bool)
                near_mask_all = np.empty(n_total, dtype=bool)
                
                for z_start in range(0, nz, chunk_z):
                    z_end = min(z_start + chunk_z, nz)
                    
                    z_range = cp.arange(z_start, z_end, dtype=cp.float64)
                    z_grid = z_range[:, None] * z_vec[None, :]
                    grid_pos = base_xy_gpu[None, :, :] + z_grid[:, None, :]
                    grid_pos = grid_pos.reshape(-1, 3)
                    
                    grid_norm = cp.sum(grid_pos ** 2, axis=1, keepdims=True)
                    dot_product = grid_pos @ atoms_pos_gpu.T
                    dists_sq = grid_norm + atoms_norm - 2 * dot_product
                    dists = cp.sqrt(cp.maximum(dists_sq, 0))
                    
                    valid_mask = cp.all(dists > radii_min_gpu[None, :], axis=1)
                    near_mask = cp.any(dists <= radii_max_gpu[None, :], axis=1)
                    
                    start_idx = z_start * n_xy
                    end_idx = z_end * n_xy
                    valid_mask_all[start_idx:end_idx] = valid_mask.get()
                    near_mask_all[start_idx:end_idx] = near_mask.get()
                    
                    del grid_pos, grid_norm, dot_product, dists_sq, dists, valid_mask, near_mask
                
                # Clean up GPU memory before returning
                del atoms_pos_gpu, radii_min_gpu, radii_max_gpu
                del base_xy_gpu, z_vec, atoms_norm
                cp.get_default_memory_pool().free_all_blocks()
                
                # Clean up pinned memory
                del atoms_pos_pinned, radii_min_pinned, radii_max_pinned
                del base_xy_pinned, z_vec_pinned
                
                return valid_mask_all, near_mask_all, False

    @staticmethod
    def _cleanup_gpu_memory(device_ids=None):
        """Clean up GPU memory after grid filtering.
        
        Args:
            device_ids: List of device IDs to clean up, or None for all used devices
        """
        try:
            import cupy as cp
            import gc
            if device_ids is None:
                device_ids = [0]  # Default
            
            # Force Python garbage collection first
            gc.collect()
            
            total_freed = 0
            for device_id in device_ids:
                with cp.cuda.Device(device_id):
                    # Get memory pool and free all blocks
                    mempool = cp.get_default_memory_pool()
                    pinned_mempool = cp.get_default_pinned_memory_pool()
                    
                    # Get memory stats before cleanup
                    used_before = mempool.used_bytes()
                    
                    # Free all blocks (including device and pinned memory)
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    
                    # Get memory stats after cleanup
                    used_after = mempool.used_bytes()
                    freed = used_before - used_after
                    total_freed += freed
                    
            # Force synchronization to ensure cleanup is complete
            cp.cuda.Device(0).synchronize()
            
            if total_freed > 0:
                print(f"  GPU memory freed: {total_freed / 1024**3:.2f} GB")
        except Exception as e:
            print(f"  Warning: GPU memory cleanup failed: {e}")
    
    @staticmethod
    def _print_gpu_memory_stats(device_ids=None, label=""):
        """Print GPU memory statistics for debugging.
        
        Args:
            device_ids: List of device IDs to check
            label: Label to print with the stats
        """
        try:
            import cupy as cp
            if device_ids is None:
                device_ids = [0]
            
            for device_id in device_ids:
                with cp.cuda.Device(device_id):
                    mempool = cp.get_default_memory_pool()
                    free_mem, total_mem = cp.cuda.Device(device_id).mem_info
                    
                    used_bytes = mempool.used_bytes()
                    total_bytes = mempool.total_bytes()
                    
                    print(f"  {label}GPU {device_id}: "
                          f"Free: {free_mem/1024**3:.2f}GB, "
                          f"Used: {used_bytes/1024**3:.2f}GB, "
                          f"Pool: {total_bytes/1024**3:.2f}GB")
        except Exception:
            pass

    def _filter_grid_gpu_multidevice(self, eff_atoms_pos, eff_radii_min, eff_radii_max,
                                      base_xy, axis, nz, nx, ny, devices):
        """
        Multi-GPU grid filtering - distributes z-layers across multiple GPUs.
        
        Args:
            devices: List of GPU device IDs to use
        """
        import cupy as cp
        from concurrent.futures import ThreadPoolExecutor
        
        n_gpus = len(devices)
        n_xy = nx * ny
        n_total = n_xy * nz
        
        # Calculate how many z-layers per GPU
        layers_per_gpu = nz // n_gpus
        extra_layers = nz % n_gpus
        
        def process_gpu_layers(args):
            device_id, z_start, z_end = args
            with cp.cuda.Device(device_id):
                n_atoms = len(eff_atoms_pos)
                
                # Use pinned memory for faster transfers
                atoms_pos_pinned = cp.cuda.alloc_pinned_memory(eff_atoms_pos.nbytes)
                atoms_pos_np = np.frombuffer(atoms_pos_pinned, dtype=eff_atoms_pos.dtype,
                                             count=eff_atoms_pos.size).reshape(eff_atoms_pos.shape)
                np.copyto(atoms_pos_np, eff_atoms_pos)
                
                radii_min_pinned = cp.cuda.alloc_pinned_memory(eff_radii_min.nbytes)
                radii_min_np = np.frombuffer(radii_min_pinned, dtype=eff_radii_min.dtype,
                                             count=eff_radii_min.size)
                np.copyto(radii_min_np, eff_radii_min)
                
                radii_max_pinned = cp.cuda.alloc_pinned_memory(eff_radii_max.nbytes)
                radii_max_np = np.frombuffer(radii_max_pinned, dtype=eff_radii_max.dtype,
                                             count=eff_radii_max.size)
                np.copyto(radii_max_np, eff_radii_max)
                
                base_xy_pinned = cp.cuda.alloc_pinned_memory(base_xy.nbytes)
                base_xy_np = np.frombuffer(base_xy_pinned, dtype=base_xy.dtype,
                                           count=base_xy.size).reshape(base_xy.shape)
                np.copyto(base_xy_np, base_xy)
                
                z_vec_pinned = cp.cuda.alloc_pinned_memory(axis[2].nbytes)
                z_vec_np = np.frombuffer(z_vec_pinned, dtype=axis[2].dtype,
                                         count=axis[2].size)
                np.copyto(z_vec_np, axis[2])
                
                # Transfer data to this GPU from pinned memory
                atoms_pos_gpu = cp.asarray(atoms_pos_np, dtype=cp.float64)
                radii_min_gpu = cp.asarray(radii_min_np, dtype=cp.float64)
                radii_max_gpu = cp.asarray(radii_max_np, dtype=cp.float64)
                base_xy_gpu = cp.asarray(base_xy_np, dtype=cp.float64)
                z_vec = cp.asarray(z_vec_np, dtype=cp.float64)
                atoms_norm = cp.sum(atoms_pos_gpu ** 2, axis=1)
                
                n_z_local = z_end - z_start
                n_points_local = n_z_local * n_xy
                
                valid_mask_local = np.empty(n_points_local, dtype=bool)
                near_mask_local = np.empty(n_points_local, dtype=bool)
                
                # Process in chunks if needed
                free_mem = cp.cuda.Device(device_id).mem_info[0]
                bytes_per_point = 24 + 8 * n_atoms * 2
                chunk_z = max(1, int((free_mem * 0.5 / bytes_per_point) / n_xy))
                
                for local_z_start in range(0, n_z_local, chunk_z):
                    local_z_end = min(local_z_start + chunk_z, n_z_local)
                    actual_z_start = z_start + local_z_start
                    actual_z_end = z_start + local_z_end
                    
                    z_range = cp.arange(actual_z_start, actual_z_end, dtype=cp.float64)
                    z_grid = z_range[:, None] * z_vec[None, :]
                    grid_pos = base_xy_gpu[None, :, :] + z_grid[:, None, :]
                    grid_pos = grid_pos.reshape(-1, 3)
                    
                    grid_norm = cp.sum(grid_pos ** 2, axis=1, keepdims=True)
                    dot_product = grid_pos @ atoms_pos_gpu.T
                    dists_sq = grid_norm + atoms_norm - 2 * dot_product
                    dists = cp.sqrt(cp.maximum(dists_sq, 0))
                    
                    valid_mask = cp.all(dists > radii_min_gpu[None, :], axis=1)
                    near_mask = cp.any(dists <= radii_max_gpu[None, :], axis=1)
                    
                    start_idx = local_z_start * n_xy
                    end_idx = local_z_end * n_xy
                    valid_mask_local[start_idx:end_idx] = valid_mask.get()
                    near_mask_local[start_idx:end_idx] = near_mask.get()
                    
                    del grid_pos, grid_norm, dot_product, dists_sq, dists, valid_mask, near_mask
                
                # Clean up GPU memory before returning from this device
                del atoms_pos_gpu, radii_min_gpu, radii_max_gpu
                del base_xy_gpu, z_vec, atoms_norm
                cp.get_default_memory_pool().free_all_blocks()
                
                # Clean up pinned memory
                del atoms_pos_pinned, radii_min_pinned, radii_max_pinned
                del base_xy_pinned, z_vec_pinned
                
                return device_id, z_start, z_end, valid_mask_local, near_mask_local
        
        # Prepare tasks for each GPU
        tasks = []
        z_offset = 0
        for i, device_id in enumerate(devices):
            n_layers = layers_per_gpu + (1 if i < extra_layers else 0)
            if n_layers > 0:
                tasks.append((device_id, z_offset, z_offset + n_layers))
                z_offset += n_layers
        
        print(f"  Distributing {nz} z-layers across {len(tasks)} GPUs: {[t[0] for t in tasks]}")
        
        # Execute on all GPUs in parallel using threads
        results = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(process_gpu_layers, task) for task in tasks]
            for future in futures:
                results.append(future.result())
        
        # Combine results in correct order
        valid_mask_all = np.empty(n_total, dtype=bool)
        near_mask_all = np.empty(n_total, dtype=bool)
        
        for device_id, z_start, z_end, valid_local, near_local in results:
            start_idx = z_start * n_xy
            end_idx = z_end * n_xy
            valid_mask_all[start_idx:end_idx] = valid_local
            near_mask_all[start_idx:end_idx] = near_local
        
        return valid_mask_all, near_mask_all

    def _filter_grid(self):
        cube = self.cube
        n_grid = cube["n_grid"]
        axis = cube["axis_vector"]
        nx, ny, nz = n_grid[0], n_grid[1], n_grid[2]
        n_total = nx * ny * nz

        atom_indices = np.array(cube["atom_index"], dtype=int) - 1
        raw_radii = np.array([VDW_RADII[idx] for idx in atom_indices], dtype=np.float64)
        vdw_radii = self.vdw_factor * raw_radii
        vdw_rmax = self.vdw_max * raw_radii
        
        atom_pos_base = np.asarray(cube["atom_pos"], dtype=np.float64)
        neigh_shifts = self._neighbor_shifts_27
        
        corners_iter = itertools.product([0, 1], repeat=3)
        box_corners = np.dot(np.array(list(corners_iter)), axis)
        box_min, box_max = np.min(box_corners, axis=0), np.max(box_corners, axis=0)
        
        global_max_r = np.max(vdw_rmax) if len(vdw_rmax) > 0 else 5.0
        safe_margin = global_max_r + 0.1
        limit_min, limit_max = box_min - safe_margin, box_max + safe_margin
        
        all_atoms_pos = (atom_pos_base[:, None, :] + neigh_shifts[None, :, :]).reshape(-1, 3)
        all_radii_min = np.repeat(vdw_radii, len(neigh_shifts))
        all_radii_max = np.repeat(vdw_rmax, len(neigh_shifts))
        
        in_bound_mask = (
            (all_atoms_pos[:, 0] >= limit_min[0]) & (all_atoms_pos[:, 0] <= limit_max[0]) &
            (all_atoms_pos[:, 1] >= limit_min[1]) & (all_atoms_pos[:, 1] <= limit_max[1]) &
            (all_atoms_pos[:, 2] >= limit_min[2]) & (all_atoms_pos[:, 2] <= limit_max[2])
        )
        
        eff_atoms_pos = all_atoms_pos[in_bound_mask]
        eff_radii_min = all_radii_min[in_bound_mask]
        eff_radii_max = all_radii_max[in_bound_mask]
        
        print(f"Grid Points: {n_total}, Effective Atoms: {len(eff_atoms_pos)}")

        j_range = np.arange(ny)
        i_range = np.arange(nx)
        j_grid, i_grid = np.meshgrid(j_range, i_range, indexing='ij')
        
        base_xy = np.zeros((nx * ny, 3), dtype=np.float64)
        base_xy[:, 0] = i_grid.flatten() * axis[0][0] + j_grid.flatten() * axis[1][0]
        base_xy[:, 1] = i_grid.flatten() * axis[0][1] + j_grid.flatten() * axis[1][1]
        base_xy[:, 2] = i_grid.flatten() * axis[0][2] + j_grid.flatten() * axis[1][2]

        # Try GPU first for large grids
        valid_mask_all = None
        near_mask_all = None
        if _GPU_CONFIG['enabled'] and _GPU_INFO['available'] and n_total >= 100000:
            devices = _GPU_CONFIG['devices']
            n_gpus = len(devices)
            
            if n_gpus == 1:
                print(f"  Using single GPU (device {devices[0]}) for grid filtering...")
            else:
                print(f"  Using {n_gpus} GPUs {devices} for grid filtering...")
            
            try:
                if n_gpus == 1:
                    # Single GPU mode
                    valid_mask_all, near_mask_all, _ = self._filter_grid_gpu_single(
                        eff_atoms_pos, eff_radii_min, eff_radii_max,
                        base_xy, axis, nz, nx, ny, device_id=devices[0]
                    )
                else:
                    # Multi-GPU mode - distribute work across devices
                    valid_mask_all, near_mask_all = self._filter_grid_gpu_multidevice(
                        eff_atoms_pos, eff_radii_min, eff_radii_max,
                        base_xy, axis, nz, nx, ny, devices
                    )
                
                # Clean up GPU memory after filtering
                if valid_mask_all is not None:
                    print(f"  Cleaning up GPU memory...")
                    self._cleanup_gpu_memory(device_ids=devices)
                    print(f"  GPU memory cleaned up successfully")
                    
            except Exception as e:
                print(f"    GPU filtering failed: {e}, falling back to CPU...")
                valid_mask_all = None
                near_mask_all = None
        
        if valid_mask_all is None:
            # Fall back to CPU processing
            MULTIPROCESS_THRESHOLD = 50000
            worker_args_template = {
                'n_grid': n_grid, 'axis': axis, 
                'atoms_pos': eff_atoms_pos, 
                'r_min': eff_radii_min, 'r_max': eff_radii_max, 
                'max_r': global_max_r, 'base_xy': base_xy
            }

            if n_total < MULTIPROCESS_THRESHOLD:
                valid_mask_all, near_mask_all = self._process_layer_chunk(
                    (0, nz, worker_args_template)
                )
            else:
                # Use optimized parallel backend
                use_threads = _NOGIL_ENABLED
                effective_cores = _get_optimal_workers(self.n_cores)
                n_workers = min(int(n_total / MULTIPROCESS_THRESHOLD),
                              max(1, min(effective_cores, nz)))
                
                chunk_size = (nz + n_workers - 1) // n_workers
                print(f"Using {n_workers} {'threads' if use_threads else 'processes'} for grid filtering (noGIL={_NOGIL_ENABLED})")
                
                tasks = []
                for i in range(n_workers):
                    z_start = i * chunk_size
                    z_end = min((i + 1) * chunk_size, nz)
                    if z_start >= z_end: 
                        break
                    tasks.append((z_start, z_end, worker_args_template))
                
                # Execute with appropriate backend
                if use_threads and _NOGIL_ENABLED:
                    # Thread-based execution for noGIL Python
                    with ThreadPoolExecutor(max_workers=n_workers) as executor:
                        futures = [executor.submit(self._process_layer_chunk, task) 
                                  for task in tasks]
                        results = [f.result() for f in futures]
                else:
                    # Process-based execution for standard Python
                    from multiprocessing import get_context
                    try:
                        ctx = get_context('spawn')
                    except:
                        ctx = get_context()
                        
                    with ctx.Pool(processes=n_workers) as pool:
                        results = pool.map(self._process_layer_chunk, tasks)
                        results = list(results)  # Convert iterator to list
                    
                valid_mask_all = np.concatenate([r[0] for r in results])
                near_mask_all = np.concatenate([r[1] for r in results])

        final_mask = valid_mask_all & near_mask_all
        flat_indices = np.flatnonzero(final_mask)
        
        V_pot_flat = cube["V_pot"].flatten(order='F')
        V_pot_filtered = V_pot_flat[final_mask]
        
        layer_size = nx * ny
        k_idx = flat_indices // layer_size
        rem = flat_indices % layer_size
        j_idx = rem // nx
        i_idx = rem % nx
        
        grid_pos_final = (
            i_idx[:, None] * axis[0] + 
            j_idx[:, None] * axis[1] + 
            k_idx[:, None] * axis[2]
        )
        
        return grid_pos_final, (V_pot_filtered - np.mean(V_pot_filtered))

    @staticmethod
    def _process_layer_chunk(args):
        z_start, z_end, params = args
        n_grid = params['n_grid']
        axis = params['axis']
        atoms_pos = params['atoms_pos']
        r_min = params['r_min']
        r_max = params['r_max']
        max_r = params['max_r']
        base_xy = params['base_xy']
        
        nx, ny = n_grid[0], n_grid[1]
        layer_size = nx * ny
        n_layers = z_end - z_start
        total_points = n_layers * layer_size
        
        chunk_valid = np.empty(total_points, dtype=bool)
        chunk_near = np.empty(total_points, dtype=bool)
        
        MEM_LIMIT_BYTES = 512 * 1024 * 1024
        n_atoms = len(atoms_pos)
        if n_atoms > 0:
            batch_size = max(1, MEM_LIMIT_BYTES // (n_atoms * 8))
        else:
            batch_size = total_points
            
        z_vec = axis[2]
        
        for start_idx in range(0, total_points, batch_size):
            end_idx = min(start_idx + batch_size, total_points)
            
            batch_indices = np.arange(start_idx, end_idx)
            k_local = batch_indices // layer_size
            xy_indices = batch_indices % layer_size
            k_global = z_start + k_local
            
            batch_coords = base_xy[xy_indices] + (k_global[:, None] * z_vec)
            
            if n_atoms == 0:
                chunk_valid[start_idx:end_idx] = True
                chunk_near[start_idx:end_idx] = False
                continue

            min_k = k_global[0]
            max_k = k_global[-1]
            z_height_min = min_k * z_vec[2]
            z_height_max = max_k * z_vec[2]
            mid_z = (z_height_min + z_height_max) / 2.0
            half_height = (z_height_max - z_height_min) / 2.0
            
            z_mask = np.abs(atoms_pos[:, 2] - mid_z) < (half_height + max_r + 1.0)
            local_atoms = atoms_pos[z_mask]
            local_rmin = r_min[z_mask]
            local_rmax = r_max[z_mask]
            
            if len(local_atoms) == 0:
                chunk_valid[start_idx:end_idx] = True
                chunk_near[start_idx:end_idx] = False
                continue
                
            dists = cdist(batch_coords, local_atoms)
            chunk_valid[start_idx:end_idx] = np.all(dists > local_rmin, axis=1)
            chunk_near[start_idx:end_idx] = np.any(dists <= local_rmax, axis=1)
            del dists

        return chunk_valid, chunk_near

    # -------------------- Fit Logic (Streaming) --------------------

    def fit_charges(self):
        """
        Fit charges using streaming Grid Chunking to avoid OOM.
        Accumulates ATA and ATb matrices incrementally.
        """
        n_atoms_orig = self.cube["n_atoms"]
        n_grid_total = len(self.grid_pos)
        
        print(f"Computing interaction matrix using {self.ewald.n_cores} cores...")
        t0 = time.time()
        
        # 1. Determine safe chunk size based on available memory
        if psutil:
            avail_mem = psutil.virtual_memory().available
        else:
            avail_mem = 4 * 1024 * 1024 * 1024 # 4GB fallback
            
        # We need to store phi_chunk (chunk_size * n_atoms * 8 bytes)
        # Allow using 40% of available RAM for the buffer
        safe_mem = avail_mem * 0.4
        bytes_per_row = n_atoms_orig * 8
        chunk_size = int(safe_mem // max(1, bytes_per_row))
        # Cap chunk size to avoid excessive process creation overhead or too large chunks
        chunk_size = max(100, min(chunk_size, 200000))
        
        print(f"  Grid total: {n_grid_total}, Chunk size: {chunk_size}")

        # 2. Initialize accumulators for Least Squares
        # A = (Phi - Phi_bar).T @ (Phi - Phi_bar)
        # b = (Phi - Phi_bar).T @ V_pot
        # We accumulate: ATA = Phi.T @ Phi, ATb = Phi.T @ V, sum_phi, sum_V
        ATA = np.zeros((n_atoms_orig, n_atoms_orig), dtype=np.float64)
        ATb = np.zeros(n_atoms_orig, dtype=np.float64)
        sum_phi = np.zeros(n_atoms_orig, dtype=np.float64)
        sum_V = 0.0
        
        # For statistics optimization: save sum_phi to avoid recomputation
        # Memory: n_atoms doubles (negligible)
        self._sum_phi = None  # Will be set at the end
        
        # Check if we have enough memory to save full Phi matrix for stats
        phi_matrix_bytes = n_grid_total * n_atoms_orig * 8
        if psutil and phi_matrix_bytes < psutil.virtual_memory().available * 0.5:
            print(f"  Saving Phi matrix for statistics ({phi_matrix_bytes/1024**3:.2f} GB)")
            self._phi_matrix = np.zeros((n_grid_total, n_atoms_orig), dtype=np.float64)
            save_phi = True
        else:
            self._phi_matrix = None
            save_phi = False
        
        # 3. Stream loop
        for start in range(0, n_grid_total, chunk_size):
            end = min(start + chunk_size, n_grid_total)
            grid_chunk = self.grid_pos[start:end]
            V_chunk = self.V_pot[start:end]
            
            # Compute partial Phi matrix
            # progress=False to avoid spamming bars for every chunk
            phi_chunk = self.ewald.compute_batch_parallel(
                grid_chunk, self.cube["atom_pos"], progress=False
            )
            
            # Accumulate
            ATA += phi_chunk.T @ phi_chunk
            ATb += phi_chunk.T @ V_chunk
            sum_phi += np.sum(phi_chunk, axis=0)
            sum_V += np.sum(V_chunk)
            
            # Save to Phi matrix if memory allows
            if save_phi:
                self._phi_matrix[start:end, :] = phi_chunk
            
            # Progress indicator
            if (start // chunk_size) % 5 == 0 or end == n_grid_total:
                print(f"    Processed {end}/{n_grid_total} grid points...", flush=True)
            
            # Explicitly free memory (only if not saved)
            if not save_phi:
                del phi_chunk

        print(f"  Ewald streaming completed in {time.time() - t0:.2f} seconds")

        # 4. Finalize A and b with mean subtraction
        # A = ATA - (1/N) * S_phi.T @ S_phi
        phi_bar = sum_phi / n_grid_total
        V_bar = sum_V / n_grid_total
        
        # Correct A: A = ATA - N * outer(phi_bar, phi_bar)
        A = ATA - n_grid_total * np.outer(phi_bar, phi_bar)
        
        # Correct b: b = ATb - sum_phi * V_bar - phi_bar * sum_V + N * phi_bar * V_bar
        # Simplifies to: b = ATb - N * phi_bar * V_bar
        b = ATb - n_grid_total * phi_bar * V_bar

        # 5. Apply Constraints & Solve
        weights = np.ones(n_atoms_orig, dtype=np.float64)
        atom_types = self.cube["atom_index"].copy()

        if self.symm_data:
            print("  Applying symmetry constraints...")
            A, b, weights, atom_types = self._apply_symmetry(A, b, weights, atom_types)
            n_atoms = len(atom_types)
        else:
            n_atoms = n_atoms_orig

        if self.resp_data:
            print("  Applying RESP restraints...")
            A, b = self._apply_resp(A, b, atom_types)
        elif self.qeq_data:
            print("  Applying QEq restraints...")
            A, b = self._apply_qeq(A, b, atom_types)

        n = n_atoms + 1
        A_solv = np.zeros((n, n), dtype=np.float64)
        A_solv[:n_atoms, :n_atoms] = A
        A_solv[:n_atoms, n_atoms] = weights
        A_solv[n_atoms, :n_atoms] = weights

        b_solv = np.zeros(n, dtype=np.float64)
        b_solv[:n_atoms] = b
        b_solv[n_atoms] = self.q_tot

        try:
            solution = linalg.solve(A_solv, b_solv)
        except np.linalg.LinAlgError:
            print("Warning: Matrix singular, using least squares fallback.")
            solution = np.linalg.lstsq(A_solv, b_solv, rcond=None)[0]
            
        charges_reduced = solution[:n_atoms]
        charges = self._expand_charges(charges_reduced) if self.symm_data else charges_reduced
        
        # Save sum_phi for statistics optimization
        # Expand if symmetry was applied
        if self.symm_data:
            # Need to expand sum_phi to full atom set
            self._sum_phi = self._expand_sum_phi(sum_phi)
        else:
            self._sum_phi = sum_phi
        self._sum_V = sum_V
        
        return charges

    def compute_stats(self, charges, need_v_coul=True):
        """
        Compute RMS error using streaming to avoid reconstructing full Phi matrix.
        
        Args:
            charges: Fitted atomic charges
            need_v_coul: If True, compute and return full V_coul array (for output file)
                        If False, only compute statistics (saves memory)
        
        Returns:
            dict with 'rms_error', 'avrg_qm', 'avrg_coul', and optionally 'V_coul'
        """
        print("Computing statistics ...")
        n_grid_total = len(self.grid_pos)
        
        # Check if we have pre-computed sum_phi from fit_charges
        # This avoids recomputing the mean
        if hasattr(self, '_sum_phi') and self._sum_phi is not None:
            # Fast path: use saved sum_phi
            avrg_coul = float(self._sum_phi @ charges) / n_grid_total
            print(f"  Using pre-computed sum_phi (fast path)")
        else:
            avrg_coul = None  # Will compute from stream
            print(f"  Warning: sum_phi not available, computing from stream")
        
        avrg_qm = np.mean(self.V_pot)
        
        # Same chunking logic
        if psutil:
            avail_mem = psutil.virtual_memory().available
        else:
            avail_mem = 4 * 1024 * 1024 * 1024
        
        bytes_per_row = self.n_atoms * 8
        chunk_size = int((avail_mem * 0.4) // max(1, bytes_per_row))
        chunk_size = max(100, min(chunk_size, 200000))
        
        # Allocate V_coul only if needed
        if need_v_coul:
            V_coul_all = np.zeros(n_grid_total, dtype=np.float64)
        else:
            V_coul_all = None
        
        # Stream computation with RMS accumulation
        sum_squared_diff = 0.0
        
        # Check if we have pre-computed Phi matrix from fit_charges
        if hasattr(self, '_phi_matrix') and self._phi_matrix is not None:
            print(f"  Using pre-computed Phi matrix (fast path)")
            phi_matrix = self._phi_matrix
            use_precomputed = True
        else:
            print(f"  Computing Phi matrix on-the-fly...")
            use_precomputed = False
        
        for start in range(0, n_grid_total, chunk_size):
            end = min(start + chunk_size, n_grid_total)
            
            if use_precomputed:
                # Fast path: directly slice from saved Phi matrix
                phi_chunk = phi_matrix[start:end, :]
            else:
                # Slow path: compute Ewald potential
                grid_chunk = self.grid_pos[start:end]
                phi_chunk = self.ewald.compute_batch_parallel(
                    grid_chunk, self.cube["atom_pos"], progress=False
                )
            
            v_coul_chunk = phi_chunk @ charges
            
            if need_v_coul:
                V_coul_all[start:end] = v_coul_chunk
            
            # Accumulate squared differences for RMS
            # V_diff = (V_coul - avrg_coul) - (V_qm - avrg_qm)
            v_qm_chunk = self.V_pot[start:end]
            if avrg_coul is not None:
                # Use pre-computed mean
                diff_chunk = (v_coul_chunk - avrg_coul) - (v_qm_chunk - avrg_qm)
            else:
                # Will need to compute mean later
                diff_chunk = v_coul_chunk - v_qm_chunk
            sum_squared_diff += np.sum(diff_chunk ** 2)
            
            del phi_chunk
        
        # Compute RMS
        if avrg_coul is None:
            # Need to compute mean from accumulated data
            # This requires another pass or storing all data
            # For now, fall back to computing mean from V_coul_all if we have it
            if V_coul_all is not None:
                avrg_coul = np.mean(V_coul_all)
                # Recompute with correct mean
                sum_squared_diff = 0.0
                for start in range(0, n_grid_total, chunk_size):
                    end = min(start + chunk_size, n_grid_total)
                    v_coul_chunk = V_coul_all[start:end]
                    v_qm_chunk = self.V_pot[start:end]
                    diff_chunk = (v_coul_chunk - avrg_coul) - (v_qm_chunk - avrg_qm)
                    sum_squared_diff += np.sum(diff_chunk ** 2)
            else:
                raise RuntimeError("Cannot compute statistics without sum_phi or V_coul_all")
        
        rms_error = float(np.sqrt(sum_squared_diff / n_grid_total))
        
        result = {
            'rms_error': rms_error,
            'avrg_qm': float(avrg_qm),
            'avrg_coul': float(avrg_coul),
        }
        
        if need_v_coul:
            result['V_coul'] = V_coul_all
        
        return result

    # -------------------- Constraints Helpers (Unchanged) --------------------

    def _apply_symmetry(self, A, b, weights, atom_types):
        trees = self.symm_data['trees']
        n_atoms_orig = self.cube["n_atoms"]

        for tree in trees:
            base = tree['base']
            for linked in tree['linked']:
                b[base] += b[linked]
                weights[base] += 1.0
                A[:, base] += A[:, linked]

        for tree in trees:
            base = tree['base']
            for linked in tree['linked']:
                A[base, :] += A[linked, :]

        linked_atoms = set()
        for tree in trees:
            linked_atoms.update(tree['linked'])
        independent = [i for i in range(n_atoms_orig) if i not in linked_atoms]

        A_reduced = A[np.ix_(independent, independent)]
        b_reduced = b[independent]
        weights_reduced = weights[independent]
        atom_types_reduced = atom_types[independent]
        return A_reduced, b_reduced, weights_reduced, atom_types_reduced

    def _expand_charges(self, charges_reduced):
        trees = self.symm_data['trees']
        n_atoms_orig = self.cube["n_atoms"]

        linked_atoms = set()
        for tree in trees:
            linked_atoms.update(tree['linked'])
        independent = [i for i in range(n_atoms_orig) if i not in linked_atoms]

        charges = np.zeros(n_atoms_orig, dtype=np.float64)
        for i, idx in enumerate(independent):
            charges[idx] = charges_reduced[i]

        for tree in trees:
            base = tree['base']
            for linked in tree['linked']:
                charges[linked] = charges[base]
        return charges

    def _expand_sum_phi(self, sum_phi_reduced):
        """Expand sum_phi from reduced set to full atom set for symmetry."""
        trees = self.symm_data['trees']
        n_atoms_orig = self.cube["n_atoms"]
        
        linked_atoms = set()
        for tree in trees:
            linked_atoms.update(tree['linked'])
        independent = [i for i in range(n_atoms_orig) if i not in linked_atoms]
        
        sum_phi = np.zeros(n_atoms_orig, dtype=np.float64)
        for i, idx in enumerate(independent):
            sum_phi[idx] = sum_phi_reduced[i]
        
        for tree in trees:
            base = tree['base']
            for linked in tree['linked']:
                sum_phi[linked] = sum_phi[base]
        return sum_phi

    def _apply_resp(self, A, b, atom_types):
        if not self.resp_data:
            return A, b
        for i, atom_type in enumerate(atom_types):
            symbol = ATOM_SYMBOLS[atom_type] if atom_type < len(ATOM_SYMBOLS) else "X"
            if symbol in self.resp_data:
                resp = self.resp_data[symbol]
                A[i, i] += resp['strength']
                b[i] += resp['strength'] * resp['target']
        return A, b

    def _apply_qeq(self, A, b, atom_types):
        for i, atom_type in enumerate(atom_types):
            symbol = ATOM_SYMBOLS[atom_type] if atom_type < len(ATOM_SYMBOLS) else "X"
            if self.qeq_data and symbol in self.qeq_data:
                qeq = self.qeq_data[symbol]
                electroneg = qeq['electroneg']
                hardness = qeq['hardness']
            elif symbol in DEFAULT_QEQ_PARAMS:
                electroneg, hardness = DEFAULT_QEQ_PARAMS[symbol]
            else:
                continue
            b[i] += electroneg
            A[i, i] += hardness
        return A, b

    # -------------------- QEq Calculation (Unchanged) --------------------

    def compute_qeq_charges(self, param_set=None):
        n_atoms = self.cube["n_atoms"]
        atom_pos = self.cube["atom_pos"]

        if param_set is None:
            param_set = DEFAULT_QEQ_PARAMS
            method_name = "QEq"
        else:
            method_name = "EQeq"

        electroneg = np.zeros(n_atoms, dtype=np.float64)
        J_idempotential = np.zeros(n_atoms, dtype=np.float64)

        h_i0 = -2.0
        h_i1 = 13.598
        eV_to_Hartree = 1.0 / 27.211386245988

        for i, atom_type in enumerate(self.cube["atom_index"]):
            symbol = ATOM_SYMBOLS[atom_type] if atom_type < len(ATOM_SYMBOLS) else "X"

            if symbol == "H" and method_name == "EQeq":
                chi_eV = 0.5 * (h_i1 + h_i0)
                J_eV = h_i1 - h_i0
                electroneg[i] = chi_eV * eV_to_Hartree
                J_idempotential[i] = J_eV * eV_to_Hartree
            elif self.qeq_data and symbol in self.qeq_data:
                electroneg[i] = self.qeq_data[symbol]['electroneg']
                J_idempotential[i] = 2.0 * self.qeq_data[symbol]['hardness']
            elif symbol in param_set:
                chi, hardness = param_set[symbol]
                electroneg[i] = chi
                J_idempotential[i] = 2.0 * hardness
            else:
                electroneg[i] = 0.15
                J_idempotential[i] = 0.50
                print(f"Warning: No parameters for {symbol}, using defaults")

        print(f"Computing {method_name} Coulomb matrix...")
        t0 = time.time()

        J_matrix = np.zeros((n_atoms, n_atoms), dtype=np.float64)
        np.fill_diagonal(J_matrix, J_idempotential)

        if self.fit_flag == 1:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    Jij = self._ewald_pairwise(atom_pos[i], atom_pos[j])
                    J_matrix[i, j] = Jij
                    J_matrix[j, i] = Jij
        else:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r_vec = atom_pos[i] - atom_pos[j]
                    r = np.linalg.norm(r_vec)
                    if r > 1e-10:
                        J_matrix[i, j] = 1.0 / r
                        J_matrix[j, i] = 1.0 / r

        print(f"  Coulomb matrix computed in {time.time() - t0:.2f} seconds")

        A = np.zeros((n_atoms, n_atoms), dtype=np.float64)
        b = np.zeros(n_atoms, dtype=np.float64)

        A[0, :] = 1.0
        b[0] = self.q_tot

        for i in range(1, n_atoms):
            A[i, :] = J_matrix[i - 1, :] - J_matrix[i, :]
            b[i] = electroneg[i] - electroneg[i - 1]

        try:
            charges = linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Warning: Matrix singular, using least squares")
            charges = np.linalg.lstsq(A, b, rcond=None)[0]

        return charges

    def _ewald_pairwise(self, pos_i, pos_j):
        delta = pos_i - pos_j
        sqrt_alpha = self.ewald.sqrt_alpha
        alpha = self.ewald.alpha
        
        J_real = 0.0
        for shift in self.ewald.shifts:
            r_vec = delta + shift
            r = np.linalg.norm(r_vec)
            if r > 1e-10:
                J_real += special.erfc(sqrt_alpha * r) / r

        J_recp = 0.0
        for kvec, kcoef in zip(self.ewald.kvecs, self.ewald.kcoefs):
            kr = np.dot(delta, kvec)
            J_recp += kcoef * np.cos(kr)

        return J_real + J_recp

def main():
    parser = argparse.ArgumentParser(
        description="REPEAT (Multi-Core Optimized with GPU and noGIL Support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance Notes:
  - GPU Acceleration: Use --gpu to enable CUDA acceleration (requires CuPy or PyTorch)
  - Python 3.13+ with noGIL: Uses ThreadPoolExecutor (shared memory, zero serialization overhead)
  - Standard Python: Uses ProcessPoolExecutor (isolated memory, higher overhead)
  - Set REPEAT_PERF=1 environment variable to enable performance monitoring
  - Use --cores to control parallel workers (default: auto-detect)

GPU Support:
  - Automatically detects CUDA-capable GPUs
  - Requires CuPy: pip install cupy-cuda11x (or appropriate version)
  - Fallback to PyTorch: pip install torch (if CuPy not available)
  - Use --gpu to enable GPU acceleration for large systems
        """
    )
    parser.add_argument("cube_file", nargs="?", default=None)
    parser.add_argument("--fit-type", type=int, default=1, choices=[0, 1])
    parser.add_argument("--vdw-factor", type=float, default=1.0)
    parser.add_argument("--vdw-max", type=float, default=1000.0)
    parser.add_argument("--cutoff", type=float, default=20.0)
    parser.add_argument("--total-charge", type=float, default=0.0)
    parser.add_argument("--symm-file", type=str, default=None)
    parser.add_argument("--resp-file", type=str, default=None)
    parser.add_argument("--qeq-file", type=str, default=None)
    parser.add_argument("--cores", type=int, default=None, help="Number of parallel workers (default: auto)")
    parser.add_argument("--block-k", type=int, default=64, help="Initial k blocking size")
    parser.add_argument("--no-jit", action="store_true", help="Disable Numba JIT")
    parser.add_argument("--gpu", nargs='?', const='0', default=None, 
                        help="Enable GPU acceleration. Optional: device IDs (e.g., --gpu, --gpu 0, --gpu 0,1,2, --gpu all)")
    parser.add_argument("--gpu-fp32", action="store_true", help="Use fp32 mixed precision for faster GPU computation")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--charge", type=str, default="repeat", choices=["repeat", "qeq"],
                        help="Charge calculation method: repeat (default) or qeq")
    parser.add_argument("--stats", action="store_true", default=False,
                        help="Compute statistics (RMS error). Disabled by default to save time.")
    args = parser.parse_args()

    # Enable GPU if requested
    if args.gpu is not None:
        devices = parse_gpu_devices(args.gpu)
        if devices:
            set_gpu_config(enabled=True, devices=devices, mixed_precision=args.gpu_fp32)
        else:
            print("\nWARNING: GPU requested but no GPU backend available.")
            print("Install CuPy: pip install cupy-cuda11x (or appropriate CUDA version)")
            print("Or PyTorch: pip install torch")
            print("Continuing with CPU...\n")

    # Print runtime info
    cpu_count = os.cpu_count() or 4
    print("="*60)
    print("REPEAT - Multi-Core Optimized")
    print("="*60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"noGIL Enabled: {_NOGIL_ENABLED}")
    print(f"CPU Cores Available: {cpu_count}")
    print(f"Optimal Workers: {_get_optimal_workers(args.cores)}")
    print(f"Numba JIT: {'Available' if _NUMBA_AVAILABLE else 'Not Available'}")
    
    if _GPU_CONFIG['enabled']:
        device_names = []
        for d in _GPU_CONFIG['devices']:
            if _GPU_INFO['cupy_available']:
                import cupy as cp
                with cp.cuda.Device(d):
                    props = cp.cuda.runtime.getDeviceProperties(d)
                    device_names.append(props['name'].decode('utf-8'))
            else:
                device_names.append(f"GPU{d}")
        precision = "mixed (fp32)" if _GPU_CONFIG['use_mixed_precision'] else "fp64"
        print(f"GPU: Enabled ({', '.join(device_names)}) [{precision}]")
    elif _GPU_INFO['available']:
        print(f"GPU: Available ({_GPU_INFO['device_name']}, use --gpu to enable)")
    else:
        print(f"GPU: Not Available")
    print("="*60)

    if (not args.cube_file) or (not os.path.exists(args.cube_file)):
        print(f"Error: Cube file '{args.cube_file}' not found!")
        sys.exit(1)

    calc = ChargeCalculator(
        cube_file=args.cube_file,
        fit_flag=args.fit_type,
        vdw_factor=args.vdw_factor,
        vdw_max=args.vdw_max,
        R_cutoff=args.cutoff,
        q_tot=args.total_charge,
        symm_file=args.symm_file,
        resp_file=args.resp_file,
        qeq_file=args.qeq_file,
        n_cores=args.cores,
        block_k=args.block_k,
        use_jit= not args.no_jit,
    )

    if args.charge == "qeq":
        print("\n========================================")
        print("Using QEq (Charge Equilibration) method")
        print("========================================")
        charges = calc.compute_qeq_charges()

        print("\nQEq Charges:")
        for i, q in enumerate(charges):
            z = calc.cube["atom_index"][i]
            sym = ATOM_SYMBOLS[z] if z < len(ATOM_SYMBOLS) else "X"
            print(f"{i+1:4d} {sym:2s} {q:12.6f}")
        print(f"\nTotal charge: {np.sum(charges):.6f}")

        if args.output:
            with open(args.output, "w") as f:
                for i, q in enumerate(charges):
                    z = calc.cube["atom_index"][i]
                    sym = ATOM_SYMBOLS[z] if z < len(ATOM_SYMBOLS) else "X"
                    pos = calc.cube["atom_pos"][i]
                    f.write(f"{i+1:4d} {sym:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f} {q:12.6f}\n")
            print(f"QEq charges written to {args.output}")

    else:
        print("\n========================================")
        print("Using REPEAT method (Streaming)")
        print("========================================")

        calc.ewald.block_k = args.block_k
        charges = calc.fit_charges()
        
        # Compute RMS stats only if explicitly requested
        # This can be time-consuming for large systems
        if args.stats or args.output:
            print("\nComputing statistics...")
            stats = calc.compute_stats(charges, need_v_coul=args.output is not None)
            rms_error = stats['rms_error']
        else:
            stats = None
            rms_error = None

        print("\nREPEAT Charges:")
        for i, q in enumerate(charges):
            z = calc.cube["atom_index"][i]
            sym = ATOM_SYMBOLS[z] if z < len(ATOM_SYMBOLS) else "X"
            print(f"{i+1:4d} {sym:2s} {q:12.6f}")
        print(f"\nTotal charge: {np.sum(charges):.6f}")
        if rms_error is not None:
            print(f"RMS error: {rms_error:.6e}")

        if args.output:
            with open(args.output, "w") as f:
                for grid_p, v_qm, v_coul in zip(calc.grid_pos, calc.V_pot, stats["V_coul"]):
                    f.write(f"{grid_p[0]:12.6f} {grid_p[1]:12.6f} {grid_p[2]:12.6f} "
                            f"{v_coul:16.8e} {v_qm:16.8e}\n")
            print(f"Output written to {args.output}")
    
    # Print performance report if enabled
    _perf_monitor.report()

if __name__ == "__main__":
    main()