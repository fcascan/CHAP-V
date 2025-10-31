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

def analyze_csv_performance_data(csv_filepath):
    """
    Analyze performance data from CSV file and return comprehensive statistics.
    Uses only standard library (no pandas required).
    
    Args:
        csv_filepath (str): Path to the CSV file containing performance metrics
        
    Returns:
        dict: Dictionary containing statistical analysis of the performance data
    """
    import csv
    import statistics
    
    try:
        # Read CSV file
        data = {}
        required_columns = ['inference_time_ms', 'total_frame_time_ms', 'cpu_usage_percent', 
                          'npu_core0_percent', 'npu_core1_percent', 'npu_core2_percent', 
                          'gpu_usage_percent', 'fps_actual', 'detections_count']
        
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            # Check required columns
            for col in required_columns:
                if col not in headers:
                    print(f"Warning: Column '{col}' not found in CSV file")
                    return None
                data[col] = []
            
            # Read data
            for row in reader:
                for col in required_columns:
                    try:
                        value = float(row[col]) if row[col] else 0.0
                        data[col].append(value)
                    except ValueError:
                        data[col].append(0.0)
        
        if not data or not data['inference_time_ms']:
            print("No valid data found in CSV file")
            return None
        
        def calc_percentile(values, percentile):
            """Calculate percentile manually"""
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n == 0:
                return 0
            index = (percentile / 100) * (n - 1)
            lower = int(index)
            upper = min(lower + 1, n - 1)
            weight = index - lower
            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
        
        def calc_stats(values):
            """Calculate basic statistics for a list of values"""
            if not values or len(values) == 0:
                return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
            
            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values)
            }
        
        # Calculate statistics
        total_frames = len(data['inference_time_ms'])
        
        # Inference time with percentiles
        inf_stats = calc_stats(data['inference_time_ms'])
        inf_stats['percentile_95'] = calc_percentile(data['inference_time_ms'], 95)
        inf_stats['percentile_99'] = calc_percentile(data['inference_time_ms'], 99)
        
        stats = {
            'total_frames': total_frames,
            'inference_time': inf_stats,
            'total_frame_time': calc_stats(data['total_frame_time_ms']),
            'cpu_usage': calc_stats(data['cpu_usage_percent']),
            'npu_usage': {
                'core0': {
                    **calc_stats(data['npu_core0_percent']),
                    'active_samples': sum(1 for x in data['npu_core0_percent'] if x > 0)
                },
                'core1': {
                    **calc_stats(data['npu_core1_percent']),
                    'active_samples': sum(1 for x in data['npu_core1_percent'] if x > 0)
                },
                'core2': {
                    **calc_stats(data['npu_core2_percent']),
                    'active_samples': sum(1 for x in data['npu_core2_percent'] if x > 0)
                }
            },
            'gpu_usage': {
                **calc_stats(data['gpu_usage_percent']),
                'active_samples': sum(1 for x in data['gpu_usage_percent'] if x > 0)
            },
            'fps': calc_stats(data['fps_actual']),
            'detections': {
                'total': sum(data['detections_count']),
                'mean_per_frame': statistics.mean(data['detections_count']) if data['detections_count'] else 0,
                'frames_with_detections': sum(1 for x in data['detections_count'] if x > 0),
                'detection_rate': (sum(1 for x in data['detections_count'] if x > 0) / total_frames * 100) if total_frames > 0 else 0
            }
        }
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing CSV file: {e}")
        return None

def print_csv_analysis(csv_filepath):
    """
    Print formatted analysis of performance data from CSV file.
    
    Args:
        csv_filepath (str): Path to the CSV file containing performance metrics
    """
    stats = analyze_csv_performance_data(csv_filepath)
    
    if stats is None:
        print("Failed to analyze CSV file")
        return
    
    print("\n" + "="*60)
    print("PERFORMANCE DATA ANALYSIS")
    print("="*60)
    print(f"Total frames analyzed: {stats['total_frames']}")
    
    print(f"\nINFERENCE TIME STATISTICS (ms)")
    print(f"  Mean: {stats['inference_time']['mean']:.2f}")
    print(f"  Median: {stats['inference_time']['median']:.2f}")
    print(f"  Std Dev: {stats['inference_time']['std']:.2f}")
    print(f"  Min: {stats['inference_time']['min']:.2f}")
    print(f"  Max: {stats['inference_time']['max']:.2f}")
    print(f"  95th percentile: {stats['inference_time']['percentile_95']:.2f}")
    print(f"  99th percentile: {stats['inference_time']['percentile_99']:.2f}")
    
    print(f"\nCPU USAGE STATISTICS (%)")
    print(f"  Mean: {stats['cpu_usage']['mean']:.1f}")
    print(f"  Median: {stats['cpu_usage']['median']:.1f}")
    print(f"  Min: {stats['cpu_usage']['min']:.1f}")
    print(f"  Max: {stats['cpu_usage']['max']:.1f}")
    
    print(f"\nNPU USAGE STATISTICS (%)")
    for core_num in range(3):
        core_key = f'core{core_num}'
        core_stats = stats['npu_usage'][core_key]
        active_percentage = (core_stats['active_samples'] / stats['total_frames']) * 100
        print(f"  NPU Core {core_num}:")
        print(f"    Mean: {core_stats['mean']:.1f}")
        print(f"    Median: {core_stats['median']:.1f}")
        print(f"    Active samples: {core_stats['active_samples']} ({active_percentage:.1f}%)")
    
    print(f"\nGPU USAGE STATISTICS (%)")
    gpu_active_percentage = (stats['gpu_usage']['active_samples'] / stats['total_frames']) * 100
    print(f"  Mean: {stats['gpu_usage']['mean']:.1f}")
    print(f"  Median: {stats['gpu_usage']['median']:.1f}")
    print(f"  Min: {stats['gpu_usage']['min']:.1f}")
    print(f"  Max: {stats['gpu_usage']['max']:.1f}")
    print(f"  Active samples: {stats['gpu_usage']['active_samples']} ({gpu_active_percentage:.1f}%)")
    
    print(f"\nFPS STATISTICS")
    print(f"  Mean: {stats['fps']['mean']:.2f}")
    print(f"  Median: {stats['fps']['median']:.2f}")
    print(f"  Min: {stats['fps']['min']:.2f}")
    print(f"  Max: {stats['fps']['max']:.2f}")
    
    print(f"\nDETECTION STATISTICS")
    print(f"  Total detections: {stats['detections']['total']}")
    print(f"  Mean detections per frame: {stats['detections']['mean_per_frame']:.2f}")
    print(f"  Frames with detections: {stats['detections']['frames_with_detections']}")
    print(f"  Detection rate: {stats['detections']['detection_rate']:.1f}%")
    
    print("="*60)

def auto_analyze_latest_csv(device_name="NPU", logger=None, csv_filepath=None):
    """
    Automatically find and analyze the most recent performance CSV file.
    
    Args:
        device_name (str): Device name to look for in CSV filename
        logger: Logger instance for output (optional)
        csv_filepath (str): Specific CSV file path to analyze (optional)
    """
    import glob
    
    def log_message(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    try:
        # Use specific file if provided, otherwise find latest
        if csv_filepath and os.path.exists(csv_filepath):
            latest_csv = csv_filepath
        else:
            # Search in both current directory and src/processing/results
            search_paths = [
                f"performance_metrics_{device_name}_*.csv",
                f"src/processing/results/performance_metrics_{device_name}_*.csv"
            ]
            
            csv_files = []
            for pattern in search_paths:
                csv_files.extend(glob.glob(pattern))
            
            if not csv_files:
                log_message("No performance CSV files found for automatic analysis")
                return
            
            # Get the most recent file
            latest_csv = max(csv_files, key=os.path.getmtime)
        
        log_message(f"Analyzing performance data from: {os.path.basename(latest_csv)}")
        
        # Generate analysis text file
        analysis_filepath = latest_csv.rsplit('.', 1)[0] + '_analysis.txt'
        
        # Capture the analysis output
        import io
        import contextlib
        
        analysis_output = io.StringIO()
        
        # Redirect print_csv_analysis output to capture it
        with contextlib.redirect_stdout(analysis_output):
            print_csv_analysis(latest_csv)
        
        analysis_text = analysis_output.getvalue()
        
        # Save analysis to file
        try:
            with open(analysis_filepath, 'w', encoding='utf-8') as f:
                f.write(f"Performance Analysis Report\n")
                f.write(f"Source CSV: {os.path.basename(latest_csv)}\n")
                f.write(f"Generated on: {time.ctime()}\n\n")
                f.write(analysis_text)
            
            log_message(f"Analysis report saved to: {os.path.basename(analysis_filepath)}")
            
            # Generate performance graphs
            try:
                from .performance_analyzer import generate_performance_graphs
                base_name = latest_csv.rsplit('.', 1)[0]
                png_path = f"{base_name}_graphs.png"
                
                generated_graph = generate_performance_graphs(latest_csv, png_path)
                if generated_graph:
                    log_message(f"Performance graphs saved to: {os.path.basename(generated_graph)}")
                else:
                    log_message("Failed to generate performance graphs")
            except ImportError:
                log_message("Cannot generate graphs - missing performance_analyzer module")
            except Exception as e:
                log_message(f"Error generating graphs: {e}")
                
        except Exception as e:
            log_message(f"Error saving analysis file: {e}")
        
        # Send output to logger if available
        if logger:
            analysis_lines = analysis_text.split('\n')
            for line in analysis_lines:
                if line.strip():  # Skip empty lines
                    logger.info(line)
        else:
            print(analysis_text)
            
    except Exception as e:
        log_message(f"Error in automatic CSV analysis: {e}")
