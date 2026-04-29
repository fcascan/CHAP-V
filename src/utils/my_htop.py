# -*- coding: utf-8 -*-
"""my_htop.py
by fcascan 2025
"""
import os
import time
import logging
import re
import threading
from ..core.config import *

SLEEP_TIME = 0.5

# Global variables for continuous monitoring
gpu_usage_samples = []
monitoring_active = False

def get_npu_info():
    try:
        npu_load_file = parser.get("MY_HTOP_PATH", "npu_load_path", fallback="/sys/kernel/debug/rknpu/load")
        npu_freq_file = parser.get("MY_HTOP_PATH", "npu_freq_path", fallback="/sys/class/devfreq/fdab0000.npu/cur_freq")
        # NPU usage
        with open(npu_load_file, "r") as f:
            data = f.read()
        percents = re.findall(r'(\d+)%', data)
        if percents:
            npu_load = [int(p) for p in percents]
        else:
            npu_load = [0, 0, 0]
        # NPU freq
        try:
            with open(npu_freq_file, "r") as f:
                npu_freq_str = f.read().strip()
            npu_freq = int(npu_freq_str) // 1000000
        except Exception:
            npu_freq = 0
        return npu_load, npu_freq
    except Exception as e:
        print(f"[ERROR] Cannot read NPU info: {e}")
        return [0, 0, 0], 0

def log_npu_usage():
    while True:
        npu_load, npu_freq = get_npu_info()
        npu_load_str = ", ".join([f"NPU_CORE_{i} Load = {load}%" for i, load in enumerate(npu_load)])
        print(f"[RKNPUTOP] {npu_load_str}, Freq = {npu_freq} MHz")
        time.sleep(SLEEP_TIME)

prev_cpu = {}
def get_cpu_info():
    cpu_loads = {}
    core_count = os.cpu_count() or 1
    global prev_cpu
    try:
        with open("/proc/stat", "r") as f:
            lines = f.readlines()
    except Exception:
        lines = []
    for i in range(core_count):
        line = next((l for l in lines if l.startswith(f"cpu{i} ")), None)
        if not line:
            continue
        parts = line.split()
        try:
            user   = int(parts[1])
            nice   = int(parts[2])
            system = int(parts[3])
            idle   = int(parts[4])
            iowait = int(parts[5])
            irq    = int(parts[6])
            softirq= int(parts[7])
            steal  = int(parts[8]) if len(parts) > 8 else 0
        except Exception:
            continue
        total = user + nice + system + idle + iowait + irq + softirq + steal
        if i in prev_cpu:
            prev_total, prev_idle = prev_cpu[i]
            diff_total = total - prev_total
            diff_idle  = idle - prev_idle
            load = (100 * (diff_total - diff_idle)) // diff_total if diff_total > 0 else 0
        else:
            load = 0
        cpu_loads[i] = load
        prev_cpu[i] = (total, idle)
    cpu_freqs = {}
    cpu_freq_pattern = parser.get("MY_HTOP_PATH", "cpu_freq_pattern", fallback="/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq")
    for i in range(core_count):
        try:
            freq_path = cpu_freq_pattern.format(core=i)
            with open(freq_path, "r") as f:
                freq_str = f.read().strip()
            freq = int(freq_str) // 1000
        except Exception:
            freq = 0
        cpu_freqs[i] = freq
    return cpu_loads, cpu_freqs

def get_gpu_info():
    gpu_load_path = parser.get("MY_HTOP_PATH", "gpu_load_path", fallback="/sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/load")
    gpu_freq_path = parser.get("MY_HTOP_PATH", "gpu_freq_path", fallback="/sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/cur_freq")
    if not os.path.exists(gpu_load_path) or not os.path.exists(gpu_freq_path):
        return None, None
    try:
        with open(gpu_load_path, "r") as f:
            raw_line = f.read().strip()
        # Format is like "15@1000000000Hz", extract the percentage
        load_str = raw_line.split('@')[0]
        gpu_load = int(load_str)
    except Exception:
        gpu_load = 0
    try:
        with open(gpu_freq_path, "r") as f:
            gpu_freq_str = f.read().strip()
        gpu_freq = int(gpu_freq_str) // 1000000
    except Exception:
        gpu_freq = 0
    return gpu_load, gpu_freq

def log_gpu_usage():
    """Continuous GPU monitoring function (similar to NPU monitoring)"""
    global gpu_usage_samples, monitoring_active
    while monitoring_active:
        gpu_load, gpu_freq = get_gpu_info()
        if gpu_load is not None:
            gpu_usage_samples.append(gpu_load)
            # Keep only last 100 samples to avoid memory issues
            if len(gpu_usage_samples) > 100:
                gpu_usage_samples.pop(0)
        time.sleep(SLEEP_TIME)

def start_gpu_monitoring():
    """Start GPU monitoring thread"""
    global monitoring_active
    monitoring_active = True
    gpu_thread = threading.Thread(target=log_gpu_usage, daemon=True)
    gpu_thread.start()
    return gpu_thread

def stop_gpu_monitoring():
    """Stop GPU monitoring"""
    global monitoring_active
    monitoring_active = False

def get_processor_usage_stats(inference_device="NPU", npu_samples=None):
    """Returns a dict with CPU, NPU, and GPU usage statistics."""
    import psutil
    stats = {}
    
    # CPU usage
    try:
        cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
        stats['cpu'] = { 'avg': cpu_percent }
    except Exception:
        stats['cpu'] = None
    
    # NPU usage
    if inference_device == "NPU":
        try:
            if npu_samples and len(npu_samples) > 0:
                # Use continuous samples collected during processing for accurate statistics
                # npu_samples contains tuples like (core0_load, core1_load, core2_load)
                all_core_samples = [[], [], []]  # 3 cores
                
                for sample in npu_samples:
                    if isinstance(sample, (tuple, list)) and len(sample) >= 3:
                        for i in range(3):
                            all_core_samples[i].append(sample[i])
                
                # Calculate average per core
                per_core_avg = []
                active_cores_total = 0
                active_cores_count = 0
                
                for i, core_samples in enumerate(all_core_samples):
                    if core_samples:
                        core_avg = sum(core_samples) / len(core_samples)
                        per_core_avg.append(round(core_avg))
                        if core_avg > 0.5:  # Only count cores with meaningful usage for overall average
                            active_cores_total += core_avg
                            active_cores_count += 1
                    else:
                        per_core_avg.append(0)

                # Calculate overall average only from active cores
                if active_cores_count > 0:
                    overall_avg = active_cores_total / active_cores_count
                else:
                    # If no cores are significantly active, show the average of all cores
                    overall_avg = sum(per_core_avg) / len(per_core_avg) if per_core_avg else 0

                stats['npu'] = {
                    'avg': round(overall_avg, 1),
                    'per_core': per_core_avg,
                    'samples': len(npu_samples)
                }
            else:
                # Fallback to single reading if no samples available
                npu_load, _ = get_npu_info()
                if npu_load:
                    # Calculate average only from active cores
                    active_cores = [load for load in npu_load if load > 0.5]
                    if active_cores:
                        avg_npu = sum(active_cores) / len(active_cores)
                    else:
                        # If no cores are significantly active, show the average of all cores
                        avg_npu = sum(npu_load) / len(npu_load)
                    stats['npu'] = { 'avg': round(avg_npu, 1), 'per_core': npu_load }
                else:
                    stats['npu'] = None
        except Exception:
            stats['npu'] = None
    else:
        stats['npu'] = None

    # GPU usage - now using averaged samples from continuous monitoring
    global gpu_usage_samples
    try:
        if len(gpu_usage_samples) > 0:
            avg_gpu = sum(gpu_usage_samples) / len(gpu_usage_samples)
            stats['gpu'] = { 'avg': avg_gpu, 'samples': len(gpu_usage_samples) }
        else:
            # Fallback to single sample if no continuous monitoring
            gpu_load, _ = get_gpu_info()
            if gpu_load is not None:
                stats['gpu'] = { 'avg': gpu_load, 'samples': 1 }
            else:
                stats['gpu'] = None
    except Exception:
        stats['gpu'] = None

    return stats
