# -*- coding: utf-8 -*-
"""csv_analysis.py
CSV performance analysis helpers for YOLO RKNN processing.
by fcascan 2026
"""

import csv
import io
import os
import time
import glob
import statistics

def save_instance_performance_data(csv_rows, results_dir, device_name, run_timestamp, label,
                                   logger=None, npu_core_id=None, model_name=None,
                                   benchmark_video=None, camera_index=None):
    """Write per-instance performance CSV and trigger graph generation.

    Args:
        csv_rows: list of dicts with per-frame metrics
        results_dir: absolute path to results directory
        device_name: e.g. "NPU"
        run_timestamp: shared timestamp string "YYYYMMDD_HHMMSS"
        label: instance identifier, e.g. "cam0" or "stream1"
        logger: optional logger; falls back to print
        npu_core_id: NPU core index used (0/1/2), or None when not applicable
        model_name: model filename, e.g. "april22_2.rknn"
        benchmark_video: video filename used in benchmark mode, e.g. "benchmark.mp4"
        camera_index: camera index used in camera mode, e.g. 0
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if not csv_rows:
        _log(f"[{label}] No performance data collected, skipping CSV export.")
        return

    os.makedirs(results_dir, exist_ok=True)
    core_part = f"_core{npu_core_id}" if npu_core_id is not None else ""
    csv_path = os.path.join(results_dir, f"{run_timestamp}_performance_metrics_{device_name}{core_part}_{label}.csv")

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        _log(f"[{label}] Performance CSV saved: {os.path.basename(csv_path)}")
    except Exception as e:
        _log(f"[{label}] Failed to write CSV: {e}")
        return

    try:
        auto_analyze_latest_csv(device_name=device_name, logger=logger, csv_filepath=csv_path,
                                npu_core_id=npu_core_id, model_name=model_name,
                                benchmark_video=benchmark_video, camera_index=camera_index,
                                inference_device=device_name)
    except Exception as e:
        _log(f"[{label}] Graph generation failed: {e}")


def analyze_csv_performance_data(csv_filepath):
    """
    Analyze performance data from a CSV file and return comprehensive statistics.

    Args:
        csv_filepath (str): Path to the CSV file containing performance metrics

    Returns:
        dict: Dictionary containing statistical analysis of the performance data
    """
    try:
        data = {}
        required_columns = [
            'inference_time_ms',
            'total_frame_time_ms',
            'cpu_usage_percent',
            'npu_core0_percent',
            'npu_core1_percent',
            'npu_core2_percent',
            'gpu_usage_percent',
            'fps_actual',
            'detections_count',
        ]

        with open(csv_filepath, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline().strip()
            if first_line.startswith('#'):
                while first_line.startswith('#'):
                    first_line = f.readline().strip()
                remaining = first_line + '\n' + f.read()
                reader = csv.DictReader(io.StringIO(remaining))
            else:
                f.seek(0)
                reader = csv.DictReader(f)

            headers = reader.fieldnames or []
            print(f"[INFO] CSV headers detected: {headers}")

            for col in required_columns:
                if col not in headers:
                    print(f"Warning: Column '{col}' not found in CSV file")
                    print(f"[INFO] Available columns: {headers}")
                    return None
                data[col] = []

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
            if not values:
                return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}

            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
            }

        total_frames = len(data['inference_time_ms'])

        inf_stats = calc_stats(data['inference_time_ms'])
        inf_stats['percentile_95'] = calc_percentile(data['inference_time_ms'], 95)
        inf_stats['percentile_99'] = calc_percentile(data['inference_time_ms'], 99)

        return {
            'total_frames': total_frames,
            'inference_time': inf_stats,
            'total_frame_time': calc_stats(data['total_frame_time_ms']),
            'cpu_usage': calc_stats(data['cpu_usage_percent']),
            'npu_usage': {
                'core0': {
                    **calc_stats(data['npu_core0_percent']),
                    'active_samples': sum(1 for x in data['npu_core0_percent'] if x > 0),
                },
                'core1': {
                    **calc_stats(data['npu_core1_percent']),
                    'active_samples': sum(1 for x in data['npu_core1_percent'] if x > 0),
                },
                'core2': {
                    **calc_stats(data['npu_core2_percent']),
                    'active_samples': sum(1 for x in data['npu_core2_percent'] if x > 0),
                },
            },
            'gpu_usage': {
                **calc_stats(data['gpu_usage_percent']),
                'active_samples': sum(1 for x in data['gpu_usage_percent'] if x > 0),
            },
            'fps': calc_stats(data['fps_actual']),
            'detections': {
                'total': sum(data['detections_count']),
                'mean_per_frame': statistics.mean(data['detections_count']) if data['detections_count'] else 0,
                'frames_with_detections': sum(1 for x in data['detections_count'] if x > 0),
                'detection_rate': (sum(1 for x in data['detections_count'] if x > 0) / total_frames * 100) if total_frames > 0 else 0,
            },
        }

    except Exception as e:
        print(f"Error analyzing CSV file: {e}")
        return None


def print_csv_analysis(csv_filepath):
    """Print formatted analysis of performance data from CSV file."""
    stats = analyze_csv_performance_data(csv_filepath)

    if stats is None:
        print("Failed to analyze CSV file")
        return

    print("\n" + "=" * 60)
    print("PERFORMANCE DATA ANALYSIS")
    print("=" * 60)
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
        print(f"  RKNPU Core {core_num}:")
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

    print("=" * 60)


def auto_analyze_latest_csv(device_name="NPU", logger=None, csv_filepath=None,
                            npu_core_id=None, model_name=None,
                            benchmark_video=None, camera_index=None, inference_device=None):
    """Automatically find and analyze the most recent performance CSV file."""

    def log_message(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    try:
        if csv_filepath and os.path.exists(csv_filepath):
            latest_csv = csv_filepath
        else:
            search_paths = [
                f"performance_metrics_{device_name}_*.csv",
                f"src/processing/results/performance_metrics_{device_name}_*.csv",
            ]

            csv_files = []
            for pattern in search_paths:
                csv_files.extend(glob.glob(pattern))

            if not csv_files:
                log_message("No performance CSV files found for automatic analysis")
                return

            latest_csv = max(csv_files, key=os.path.getmtime)

        log_message(f"Analyzing performance data from: {os.path.basename(latest_csv)}")

        analysis_filepath = latest_csv.rsplit('.', 1)[0] + '_analysis.txt'

        import contextlib

        analysis_output = io.StringIO()
        with contextlib.redirect_stdout(analysis_output):
            print_csv_analysis(latest_csv)

        analysis_text = analysis_output.getvalue()

        try:
            with open(analysis_filepath, 'w', encoding='utf-8') as f:
                f.write("Performance Analysis Report\n")
                f.write(f"Source CSV: {os.path.basename(latest_csv)}\n")
                f.write(f"Generated on: {time.ctime()}\n")
                if model_name:
                    f.write(f"Model: {model_name}\n")
                if npu_core_id is not None:
                    f.write(f"NPU Core: {npu_core_id}\n")
                if inference_device:
                    f.write(f"Device: {inference_device}\n")
                if benchmark_video:
                    f.write(f"Video: {benchmark_video}\n")
                if camera_index is not None:
                    f.write(f"Camera: {camera_index}\n")
                f.write("\n")
                f.write(analysis_text)

            log_message(f"Analysis report saved to: {os.path.basename(analysis_filepath)}")

            try:
                from .performance_analyzer import generate_performance_graphs

                base_name = latest_csv.rsplit('.', 1)[0]
                png_path = f"{base_name}_graphs.png"

                generated_graph = generate_performance_graphs(latest_csv, png_path,
                                                              npu_core_id=npu_core_id,
                                                              model_name=model_name,
                                                              benchmark_video=benchmark_video,
                                                              camera_index=camera_index,
                                                              inference_device=inference_device)
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

        if logger:
            for line in analysis_text.split('\n'):
                if line.strip():
                    logger.info(line)
        else:
            print(analysis_text)

    except Exception as e:
        log_message(f"Error in automatic CSV analysis: {e}")