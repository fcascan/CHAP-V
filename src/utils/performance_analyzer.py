#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
"""performance_analyzer.py
Performance analysis and visualization module for CHAP-V processing metrics
by fcascan 2026
"""

import sys
import os
import time
import argparse
import glob
import csv
import statistics
import io
from datetime import datetime

# Try to import PIL for graph generation
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError as e:
    PIL_AVAILABLE = False
    print(f"[ERROR] PIL not available: {e}")

# Statistical functions
def mean(data):
    return statistics.mean(data) if data else 0

def median(data):
    return statistics.median(data) if data else 0

def std(data):
    return statistics.stdev(data) if len(data) > 1 else 0

def percentile(data, p):
    if not data:
        return 0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]

def draw_dashed_line(draw, start, end, fill, dash_length=5):
    """Draw a dashed line using PIL"""
    x1, y1 = start
    x2, y2 = end
    
    dx = x2 - x1
    dy = y2 - y1
    distance = (dx**2 + dy**2)**0.5
    
    if distance == 0:
        return
    
    steps = int(distance / dash_length)
    for i in range(0, steps, 2):
        start_x = x1 + (dx / steps) * i * dash_length / dash_length
        start_y = y1 + (dy / steps) * i * dash_length / dash_length
        end_x = x1 + (dx / steps) * min(i + 1, steps) * dash_length / dash_length
        end_y = y1 + (dy / steps) * min(i + 1, steps) * dash_length / dash_length
        draw.line([start_x, start_y, end_x, end_y], fill=fill, width=2)

def calculate_time_axis_intervals(total_points):
    """Calculate appropriate time axis intervals based on total duration"""
    if total_points < 60:  # Less than 60 points (1 second per point at 60fps ≈ 1 second)
        return 1, "1s"  # Every 1 second
    elif total_points < 3600:  # Less than 60 minutes
        return max(1, total_points // 60), "1min"  # Every 1 minute
    else:  # More than 60 minutes
        return max(1, total_points // 60), "1hr"  # Every 1 hour equivalent


def _nice_time_ticks(duration_s, target=10):
    """Dynamic, integer-valued time-axis ticks spanning 0..duration_s.
    Returns [(t_seconds, "<int><unit>"), ...] with ~`target` equidistant marks. The step is snapped to a
    time-friendly ladder (1/2/5/10/15/30 · s, min, h …) and the unit (s/m/h) is DERIVED from the chosen
    step, so labels are ALWAYS whole numbers and the axis scales from a few seconds to many hours.
    Prefers integer tick values over hitting exactly `target` marks (per user preference)."""
    if not duration_s or duration_s <= 0:
        return []
    ladder = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800,
              3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800]
    step = next((s for s in ladder if duration_s / s <= target), None)
    if step is None:  # longer than the ladder -> round up to a whole number of days
        step = (int(duration_s // (target * 86400)) + 1) * 86400
    if step % 3600 == 0:
        unit, f = 'h', 3600
    elif step % 60 == 0:
        unit, f = 'm', 60
    else:
        unit, f = 's', 1
    ticks, t = [], 0
    while t <= duration_s + 1e-6:
        ticks.append((t, f"{t // f}{unit}"))
        t += step
    return ticks


def _draw_time_axis(draw, gx, gw, y_bottom, duration_s, ticks, font):
    """Draw x-axis time ticks at their TRUE time fraction (x = gx + t/duration·gw), skipping 0.
    Shared by every report graph so they all use one dynamic, integer-valued axis that fits the run
    duration (fixes the old per-index labels that collapsed to '0min' on short runs)."""
    if not duration_s or duration_s <= 0:
        return
    for t, label in ticks:
        if t <= 0 or t > duration_s:
            continue
        x = gx + int(t / duration_s * gw)
        draw.line([x, y_bottom, x, y_bottom + 5], fill='black', width=1)
        draw.text((x - 12, y_bottom + 8), label, fill='gray', font=font)

def generate_performance_reports(csv_filepath, output_path=None, npu_core_id=None, model_name=None,
                                benchmark_video=None, camera_index=None, inference_device=None):
    """
    Generate performance analysis reports as PNG file using PIL.

    Args:
        csv_filepath (str): Path to the CSV file
        output_path (str): Output path for the PNG file (optional)
        npu_core_id (int|None): NPU core index used for this run (0/1/2), or None
        model_name (str|None): Model filename, e.g. "detectS2.rknn"
        benchmark_video (str|None): Video filename used in benchmark mode
        camera_index (int|None): Camera index used in camera mode
        inference_device (str|None): Inference device, e.g. "NPU"

    Returns:
        str: Path to the generated PNG file
    """
    if not PIL_AVAILABLE:
        print("PIL not available for report generation")
        return None
        
    try:
        # Determine output path
        if output_path is None:
            base_name = os.path.splitext(csv_filepath)[0]
            output_path = f"{base_name}_report.png"
        
        # Read CSV data
        data = {}
        metadata = {}
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            # Try to read metadata from comments at the top
            first_line = f.readline().strip()
            if first_line.startswith('#'):
                # Parse metadata lines
                while first_line.startswith('#'):
                    # Extract key=value pairs from comments
                    comment_content = first_line[1:].strip()
                    if '=' in comment_content:
                        key, value = comment_content.split('=', 1)
                        metadata[key.strip()] = value.strip()
                    first_line = f.readline().strip()
                # Reset to read as DictReader
                f.seek(0)
                # Skip metadata lines
                while True:
                    line = f.readline().strip()
                    if not line.startswith('#'):
                        break
                # Create reader from current position
                remaining = line + '\n' + f.read()
                reader = csv.DictReader(io.StringIO(remaining))
            else:
                f.seek(0)
                reader = csv.DictReader(f)
            
            headers = reader.fieldnames
            print(f"[INFO] CSV headers detected: {headers}")
            
            # Initialize data lists
            for header in headers:
                data[header] = []
            
            # Read all data
            for row in reader:
                for header in headers:
                    try:
                        if header in ['timestamp']:
                            data[header].append(row[header])
                        else:
                            data[header].append(float(row[header]) if row[header] else 0.0)
                    except ValueError:
                        data[header].append(0.0)
        
        # Sanity check: required columns must be present
        required_cols = ['inference_time_ms', 'fps_actual', 'npu_core0_percent', 'cpu_usage_percent', 'frame_number']
        missing_cols = [c for c in required_cols if c not in data]
        if missing_cols:
            print(f"[ERROR] Missing columns in CSV: {missing_cols}. Available columns: {list(data.keys())}")
            return None

        if not data or len(data.get('frame_number', [])) == 0:
            print(f"No valid data found in {csv_filepath}")
            return None

        # Create image with more space for new graphs and labels
        img_width = 1400
        # Make the image height follow A4 aspect ratio (210x297 mm) based on width
        img_height = int(img_width * 297 / 210)
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font_large = ImageFont.load_default()
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Extract key metrics
        inference_times = data['inference_time_ms']
        fps_data = data['fps_actual']
        npu_core0 = data['npu_core0_percent']
        cpu_usage = data['cpu_usage_percent']
        # GPU/Hailo usage are present in every mode's CSV; tolerate older files that lack them.
        gpu_usage = data.get('gpu_usage_percent', [0.0] * len(data['frame_number']))
        hailo_usage = data.get('hailo_usage_percent', [0.0] * len(data['frame_number']))
        hailo_infer_ms = data.get('hailo_infer_ms', [0.0] * len(data['frame_number']))
        hailo_temp = data.get('hailo_temp_c', [0.0] * len(data['frame_number']))
        hailo_power = data.get('hailo_power_w', [0.0] * len(data['frame_number']))
        detections = data.get('detections_count', [0.0] * len(data['frame_number']))
        detection_frames = sum(1 for d in detections if float(d) > 0)

        # Calculate statistics
        inf_mean = mean(inference_times)
        inf_median = median(inference_times)
        inf_95th = percentile(inference_times, 95)
        fps_mean = mean(fps_data)
        npu_mean = mean(npu_core0)
        cpu_mean = mean(cpu_usage)
        gpu_mean = mean(gpu_usage)
        hailo_mean = mean(hailo_usage) if hailo_usage else 0.0
        # Mean device inference latency (ms) over frames that actually ran on the Hailo (infer_ms>0).
        _hlat = [v for v in hailo_infer_ms if v and v > 0]
        hailo_lat_mean = mean(_hlat) if _hlat else 0.0
        # Mean chip temperature / power over frames with a sample (cached from the 500 ms monitor).
        _htemp = [v for v in hailo_temp if v and v > 0]
        _hpower = [v for v in hailo_power if v and v > 0]
        hailo_temp_mean = mean(_htemp) if _htemp else 0.0
        hailo_power_mean = mean(_hpower) if _hpower else 0.0
        
        # Draw title
        title = f"Performance Analysis Report"
        subtitle = f"Source: {os.path.basename(csv_filepath)}"
        draw.text((20, 20), title, fill='black', font=font_large)
        draw.text((20, 50), subtitle, fill='black', font=font)
        draw.text((20, 75), f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fill='black', font=font)
        
        # Metadata lines
        y_meta = 95

        if benchmark_video:
            draw.text((20, y_meta), f"Video source: {benchmark_video}", fill='black', font=font)
            y_meta += 18
        elif camera_index is not None:
            draw.text((20, y_meta), f"Camera: {camera_index}", fill='black', font=font)
            y_meta += 18

        if inference_device:
            draw.text((20, y_meta), f"Device: {inference_device}", fill='black', font=font)
            y_meta += 18

        if npu_core_id is not None:
            draw.text((20, y_meta), f"RKNPU Core: {npu_core_id}", fill='black', font=font)
            y_meta += 18
        
        _model_display = model_name or (os.path.basename(metadata['model_path']) if 'model_path' in metadata else None)
        if _model_display:
            draw.text((20, y_meta), f"Model: {_model_display}", fill='black', font=font)
            y_meta += 18

        # Statistics summary
        y_pos = y_meta + 10
        stats_text = [
            f"Summary Statistics",
            f"  • Total frames analyzed: {len(inference_times)}",
            f"  • Frames with detections: {detection_frames}",
            f"  • Average inference time: {inf_mean:.1f} ms",
            f"  • Median inference time: {inf_median:.1f} ms", 
            f"  • 95th percentile: {inf_95th:.1f} ms",
            f"  • Average FPS: {fps_mean:.1f}",
            f"  • Average RKNPU usage: {npu_mean:.1f}%",
            f"  • Average Hailo-8 usage: {hailo_mean:.1f}%",
            f"  • Average CPU usage: {cpu_mean:.1f}%",
            f"  • Average GPU usage: {gpu_mean:.1f}%",
            f"  • Min inference time: {min(inference_times):.1f} ms",
            f"  • Max inference time: {max(inference_times):.1f} ms"
        ]
        
        for i, text in enumerate(stats_text):
            if i == 0:  # Header
                draw.text((20, y_pos), text, fill='black', font=font_large)
            else:
                draw.text((20, y_pos), text, fill='black', font=font)
            y_pos += 25
        
        # Calculate time axis intervals
        total_points = len(inference_times)
        interval_step, interval_label = calculate_time_axis_intervals(total_points)

        # Graph 1: Inference Time Timeline (placed after summary with padding)
        graph_y = y_pos + 40
        graph_width = 650
        graph_height = 180
        # Dynamic time axis shared by every graph; computed from real duration once timestamps are
        # parsed below (initialized here so it's always defined even if that block is skipped).
        duration_s = 0.0
        time_ticks = []
        
        # Draw inference time graph
        draw.rectangle([20, graph_y, 20 + graph_width, graph_y + graph_height], outline='black', width=2)
        draw.text((20, graph_y - 25), "Inference Time Over Time (ms)", fill='black', font=font_large)
        
        if len(inference_times) > 0:
            max_val = max(inference_times)
            min_val = min(inference_times)
            val_range = max_val - min_val if max_val != min_val else 1
            
            # Sample data points if too many
            sample_size = min(300, len(inference_times))
            step = len(inference_times) // sample_size if sample_size > 0 else 1
            sampled_data = inference_times[::step][:sample_size]
            
            # Draw grid lines with time intervals
            for i in range(5):
                y_grid = graph_y + (i * graph_height // 4)
                draw.line([20, y_grid, 20 + graph_width, y_grid], fill='lightgray', width=1)
                val = max_val - (i * val_range / 4)
                draw.text((25, y_grid - 8), f"{val:.1f}", fill='gray', font=small_font)
            
            # Parse timestamps once and prepare sampled indices for time axis labels
            timestamps_raw = data.get('timestamp', [])
            parsed_ts = []
            first_ts = None
            last_ts = None
            if timestamps_raw:
                for t in timestamps_raw:
                    try:
                        parsed = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
                    except Exception:
                        try:
                            parsed = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            parsed = None
                    parsed_ts.append(parsed)
                # find first and last valid timestamps
                for p in parsed_ts:
                    if p is not None:
                        first_ts = p
                        break
                for p in reversed(parsed_ts):
                    if p is not None:
                        last_ts = p
                        break

            # Choose label unit based on actual duration
            total_seconds = None
            unit = 's'
            factor = 1
            if first_ts and last_ts and last_ts > first_ts:
                total_seconds = (last_ts - first_ts).total_seconds()
                if total_seconds < 60:
                    unit = 's'; factor = 1
                elif total_seconds < 3600:
                    unit = 'm'; factor = 60
                else:
                    unit = 'h'; factor = 3600

            # Real run duration for the dynamic time axis: timestamps first; fall back to cumulative
            # frame time, then frame count @30 fps. All graphs reuse `time_ticks` (see _nice_time_ticks).
            duration_s = total_seconds if total_seconds else 0.0
            if not duration_s or duration_s <= 0:
                _tft = data.get('total_frame_time_ms', [])
                duration_s = (sum(_tft) / 1000.0) if _tft else (len(inference_times) / 30.0)
            time_ticks = _nice_time_ticks(duration_s)

            _draw_time_axis(draw, 20, graph_width, graph_y + graph_height, duration_s, time_ticks, small_font)
            
            if len(sampled_data) > 1:
                for i in range(len(sampled_data) - 1):
                    x1 = 20 + (i * graph_width // len(sampled_data))
                    y1 = graph_y + graph_height - int((sampled_data[i] - min_val) / val_range * graph_height)
                    x2 = 20 + ((i + 1) * graph_width // len(sampled_data))
                    y2 = graph_y + graph_height - int((sampled_data[i + 1] - min_val) / val_range * graph_height)
                    draw.line([x1, y1, x2, y2], fill='blue', width=2)
                
                # Draw mean line (dashed)
                mean_y = graph_y + graph_height - int((inf_mean - min_val) / val_range * graph_height)
                draw_dashed_line(draw, (20, mean_y), (20 + graph_width, mean_y), fill='blue')

        # Add text label below Inference Time graph
        draw.text((25, graph_y + graph_height + 20), f"Mean: {inf_mean:.1f}ms", fill='blue', font=font)
        
        # Graph 2: FPS Timeline with improved scaling
        graph2_x = 720
        draw.rectangle([graph2_x, graph_y, graph2_x + graph_width, graph_y + graph_height], outline='black', width=2)
        draw.text((graph2_x, graph_y - 25), "FPS Over Time", fill='black', font=font_large)
        
        if len(fps_data) > 0:
            max_fps = max(fps_data)
            min_fps = min(fps_data) 
            
            # Center on average, with bounds
            fps_lower_bound = max(0, min_fps - 10)
            fps_upper_bound = max_fps + 10
            fps_range = fps_upper_bound - fps_lower_bound if fps_upper_bound != fps_lower_bound else 1
            
            sampled_fps = fps_data[::step][:sample_size]
            
            # Draw grid lines
            for i in range(5):
                y_grid = graph_y + (i * graph_height // 4)
                draw.line([graph2_x, y_grid, graph2_x + graph_width, y_grid], fill='lightgray', width=1)
                val = fps_upper_bound - (i * fps_range / 4)
                draw.text((graph2_x + 5, y_grid - 8), f"{val:.1f}", fill='gray', font=small_font)
            
            # Dynamic time axis (shared helper)
            _draw_time_axis(draw, graph2_x, graph_width, graph_y + graph_height, duration_s, time_ticks, small_font)
            
            if len(sampled_fps) > 1:
                for i in range(len(sampled_fps) - 1):
                    x1 = graph2_x + (i * graph_width // len(sampled_fps))
                    y1 = graph_y + graph_height - int((sampled_fps[i] - fps_lower_bound) / fps_range * graph_height)
                    x2 = graph2_x + ((i + 1) * graph_width // len(sampled_fps))
                    y2 = graph_y + graph_height - int((sampled_fps[i + 1] - fps_lower_bound) / fps_range * graph_height)
                    draw.line([x1, y1, x2, y2], fill='green', width=2)
                
                # Draw mean line (dashed, green, and moved text below graph)
                mean_y = graph_y + graph_height - int((fps_mean - fps_lower_bound) / fps_range * graph_height)
                draw_dashed_line(draw, (graph2_x, mean_y), (graph2_x + graph_width, mean_y), fill='green')
        
        # Add text label below FPS graph
        draw.text((graph2_x + 5, graph_y + graph_height + 20), f"Mean: {fps_mean:.1f} FPS", fill='green', font=font)
        
        # spacing between rows (increased for clearer separation)
        graph_row_sep = 100

        # Graph 4: Individual NPU Usage (second row, left)
        graph4_y = graph_y + graph_height + graph_row_sep
        graph4_x = 20
        graph4_width = (img_width - 60) // 2  # two columns with 20px margins and 20px gap
        graph4_height = 140
        draw.rectangle([graph4_x, graph4_y, graph4_x + graph4_width, graph4_y + graph4_height], outline='black', width=2)
        _npu_title = f"RKNPU Core {npu_core_id} Usage (%)" if npu_core_id is not None else "RKNPU Usage (%)"
        draw.text((graph4_x, graph4_y - 25), _npu_title, fill='black', font=font_large)
        
        if len(npu_core0) > 0:
            sampled_npu = npu_core0[::step][:sample_size]
            
            # Draw grid lines
            for i in range(5):
                y_grid = graph4_y + (i * graph4_height // 4)
                draw.line([graph4_x, y_grid, graph4_x + graph4_width, y_grid], fill='lightgray', width=1)
                val = 100 - (i * 25)
                draw.text((graph4_x + 5, y_grid - 8), f"{val}%", fill='gray', font=small_font)
            
            # Dynamic time axis (shared helper)
            _draw_time_axis(draw, graph4_x, graph4_width, graph4_y + graph4_height, duration_s, time_ticks, small_font)
            
            if len(sampled_npu) > 1:
                # NPU line
                for i in range(len(sampled_npu) - 1):
                    x1 = graph4_x + (i * graph4_width // len(sampled_npu))
                    y1 = graph4_y + graph4_height - int(sampled_npu[i] * graph4_height / 100)
                    x2 = graph4_x + ((i + 1) * graph4_width // len(sampled_npu))
                    y2 = graph4_y + graph4_height - int(sampled_npu[i + 1] * graph4_height / 100)
                    draw.line([x1, y1, x2, y2], fill='orange', width=2)
                
                # Draw dashed average line
                npu_avg_y = graph4_y + graph4_height - int(npu_mean * graph4_height / 100)
                draw_dashed_line(draw, (graph4_x, npu_avg_y), (graph4_x + graph4_width, npu_avg_y), fill='darkorange')
        
        # Graph 5: Individual CPU Usage (second row, right)
        graph5_y = graph4_y
        graph5_x = graph4_x + graph4_width + 20
        graph5_width = graph4_width
        graph5_height = 140
        draw.rectangle([graph5_x, graph5_y, graph5_x + graph5_width, graph5_y + graph5_height], outline='black', width=2)
        draw.text((graph5_x, graph5_y - 25), "CPU Usage (%)", fill='black', font=font_large)
        
        if len(cpu_usage) > 0:
            sampled_cpu = cpu_usage[::step][:sample_size]
            
            # Draw grid lines
            for i in range(5):
                y_grid = graph5_y + (i * graph5_height // 4)
                draw.line([graph5_x, y_grid, graph5_x + graph5_width, y_grid], fill='lightgray', width=1)
                val = 100 - (i * 25)
                draw.text((graph5_x + 5, y_grid - 8), f"{val}%", fill='gray', font=small_font)
            
            # Dynamic time axis (shared helper)
            _draw_time_axis(draw, graph5_x, graph5_width, graph5_y + graph5_height, duration_s, time_ticks, small_font)
            
            if len(sampled_cpu) > 1:
                # CPU line
                for i in range(len(sampled_cpu) - 1):
                    x1 = graph5_x + (i * graph5_width // len(sampled_cpu))
                    y1 = graph5_y + graph5_height - int(sampled_cpu[i] * graph5_height / 100)
                    x2 = graph5_x + ((i + 1) * graph5_width // len(sampled_cpu))
                    y2 = graph5_y + graph5_height - int(sampled_cpu[i + 1] * graph5_height / 100)
                    draw.line([x1, y1, x2, y2], fill='purple', width=2)
                
                # Draw dashed average line
                cpu_avg_y = graph5_y + graph5_height - int(cpu_mean * graph5_height / 100)
                draw_dashed_line(draw, (graph5_x, cpu_avg_y), (graph5_x + graph5_width, cpu_avg_y), fill='darkviolet')
        
        # Add text labels below graphs for means
        draw.text((graph4_x + 5, graph4_y + graph4_height + 30), f"Avg: {npu_mean:.1f}%", fill='darkorange', font=font)
        draw.text((graph5_x + 5, graph5_y + graph5_height + 30), f"Avg: {cpu_mean:.1f}%", fill='darkviolet', font=font)
        
        # Graph 3 slot: Hailo Usage (%) individual graph (third row, RIGHT). The cross-backend
        # comparison now lives full-width at the bottom (see "Comparison (%)" below).
        graph3_y = graph4_y + graph4_height + graph_row_sep
        graph3_height = 150
        graph3_x = 720

        draw.rectangle([graph3_x, graph3_y, graph3_x + graph_width, graph3_y + graph3_height], outline='black', width=2)
        draw.text((graph3_x, graph3_y - 25), "Hailo Occupancy (%)", fill='black', font=font_large)

        if len(hailo_usage) > 0:
            sampled_hailo = hailo_usage[::step][:sample_size]

            # Draw grid lines
            for i in range(5):
                y_grid = graph3_y + (i * graph3_height // 4)
                draw.line([graph3_x, y_grid, graph3_x + graph_width, y_grid], fill='lightgray', width=1)
                val = 100 - (i * 25)
                draw.text((graph3_x + 5, y_grid - 8), f"{val}%", fill='gray', font=small_font)

            # Dynamic time axis (shared helper)
            _draw_time_axis(draw, graph3_x, graph_width, graph3_y + graph3_height, duration_s, time_ticks, small_font)

            if len(sampled_hailo) > 1:
                for i in range(len(sampled_hailo) - 1):
                    x1 = graph3_x + (i * graph_width // len(sampled_hailo))
                    y1 = graph3_y + graph3_height - int(sampled_hailo[i] * graph3_height / 100)
                    x2 = graph3_x + ((i + 1) * graph_width // len(sampled_hailo))
                    y2 = graph3_y + graph3_height - int(sampled_hailo[i + 1] * graph3_height / 100)
                    draw.line([x1, y1, x2, y2], fill='teal', width=2)

                hailo_avg_y = graph3_y + graph3_height - int(hailo_mean * graph3_height / 100)
                draw_dashed_line(draw, (graph3_x, hailo_avg_y), (graph3_x + graph_width, hailo_avg_y), fill='teal')

        _hailo_avg_txt = f"Avg: {hailo_mean:.1f}%  |  Latency: {hailo_lat_mean:.1f} ms"
        if hailo_temp_mean > 0:
            _hailo_avg_txt += f"  |  Temp: {hailo_temp_mean:.1f}°C"
        if hailo_power_mean > 0:
            _hailo_avg_txt += f"  |  Power: {hailo_power_mean:.2f} W"
        draw.text((graph3_x + 5, graph3_y + graph3_height + 30), _hailo_avg_txt, fill='teal', font=font)

                # Draw dashed average lines
                # npu_avg_y = graph3_y + graph3_height - int(npu_mean * graph3_height / 100)
                # cpu_avg_y = graph3_y + graph3_height - int(cpu_mean * graph3_height / 100)
                # draw_dashed_line(draw, (20, npu_avg_y), (20 + graph_width, npu_avg_y), fill='darkorange')
                # draw_dashed_line(draw, (20, cpu_avg_y), (20 + graph_width, cpu_avg_y), fill='darkviolet')
        
        # Add text labels below NPU vs CPU graph
        # draw.text((25, graph3_y + graph3_height + 20), f"NPU Avg: {npu_mean:.1f}%", fill='darkorange', font=font)
        # draw.text((25, graph3_y + graph3_height + 40), f"CPU Avg: {cpu_mean:.1f}%", fill='darkviolet', font=font)

        # Graph 6: GPU Usage (third row, LEFT) — shown for ALL modes from gpu_usage_percent.
        # In GPU-OpenCV-OpenCL runs this is the Mali load; in NPU/CPU runs it stays near 0 (expected).
        graph6_x = 20
        graph6_y = graph3_y
        graph6_width = graph_width
        graph6_height = graph3_height
        draw.rectangle([graph6_x, graph6_y, graph6_x + graph6_width, graph6_y + graph6_height], outline='black', width=2)
        draw.text((graph6_x, graph6_y - 25), "GPU Usage (%)", fill='black', font=font_large)

        if len(gpu_usage) > 0:
            sampled_gpu = gpu_usage[::step][:sample_size]

            # Grid lines
            for i in range(5):
                y_grid = graph6_y + (i * graph6_height // 4)
                draw.line([graph6_x, y_grid, graph6_x + graph6_width, y_grid], fill='lightgray', width=1)
                val = 100 - (i * 25)
                draw.text((graph6_x + 5, y_grid - 8), f"{val}%", fill='gray', font=small_font)

            # Dynamic time axis (shared helper)
            _draw_time_axis(draw, graph6_x, graph6_width, graph6_y + graph6_height, duration_s, time_ticks, small_font)

            if len(sampled_gpu) > 1:
                for i in range(len(sampled_gpu) - 1):
                    x1 = graph6_x + (i * graph6_width // len(sampled_gpu))
                    y1 = graph6_y + graph6_height - int(sampled_gpu[i] * graph6_height / 100)
                    x2 = graph6_x + ((i + 1) * graph6_width // len(sampled_gpu))
                    y2 = graph6_y + graph6_height - int(sampled_gpu[i + 1] * graph6_height / 100)
                    draw.line([x1, y1, x2, y2], fill='red', width=2)

                gpu_avg_y = graph6_y + graph6_height - int(gpu_mean * graph6_height / 100)
                draw_dashed_line(draw, (graph6_x, gpu_avg_y), (graph6_x + graph6_width, gpu_avg_y), fill='darkred')

        draw.text((graph6_x + 5, graph6_y + graph6_height + 30), f"Avg: {gpu_mean:.1f}%", fill='darkred', font=font)

        # Comparison (%) — full-width bottom timeline overlaying CPU / RKNPU / GPU / Hailo usage.
        cmp_y = graph3_y + graph3_height + graph_row_sep
        cmp_x = 20
        cmp_width = img_width - 40
        cmp_height = 320
        draw.rectangle([cmp_x, cmp_y, cmp_x + cmp_width, cmp_y + cmp_height], outline='black', width=2)
        draw.text((cmp_x, cmp_y - 25), "Comparison (%)", fill='black', font=font_large)

        for i in range(5):
            y_grid = cmp_y + (i * cmp_height // 4)
            draw.line([cmp_x, y_grid, cmp_x + cmp_width, y_grid], fill='lightgray', width=1)
            val = 100 - (i * 25)
            draw.text((cmp_x + 5, y_grid - 8), f"{val}%", fill='gray', font=small_font)

        _ref = cpu_usage if len(cpu_usage) > 0 else npu_core0
        if len(_ref) > 0:
            # Dynamic time axis (shared helper)
            _draw_time_axis(draw, cmp_x, cmp_width, cmp_y + cmp_height, duration_s, time_ticks, small_font)

            # Overlay one line per backend (0% when a backend is not the active one).
            for _series, _color in ((cpu_usage, 'purple'), (npu_core0, 'orange'), (gpu_usage, 'red'), (hailo_usage, 'teal')):
                _samp = _series[::step][:sample_size]
                if len(_samp) < 2:
                    continue
                for i in range(len(_samp) - 1):
                    x1 = cmp_x + (i * cmp_width // len(_samp))
                    y1 = cmp_y + cmp_height - int(_samp[i] * cmp_height / 100)
                    x2 = cmp_x + ((i + 1) * cmp_width // len(_samp))
                    y2 = cmp_y + cmp_height - int(_samp[i + 1] * cmp_height / 100)
                    draw.line([x1, y1, x2, y2], fill=_color, width=2)

        # Legend (below the Comparison chart)
        legend_x = 20
        legend_y = cmp_y + cmp_height + 25
        draw.text((legend_x, legend_y), "LEGEND:", fill='black', font=font_large)
        
        # Solid lines
        draw.line([legend_x, legend_y + 30, legend_x + 30, legend_y + 30], fill='blue', width=3)
        draw.text((legend_x + 40, legend_y + 25), "Inference Time (ms)", fill='black', font=font)
        
        draw.line([legend_x, legend_y + 50, legend_x + 30, legend_y + 50], fill='green', width=3)
        draw.text((legend_x + 40, legend_y + 45), "FPS Over Time", fill='black', font=font)
        
        draw.line([legend_x, legend_y + 70, legend_x + 30, legend_y + 70], fill='orange', width=3)
        draw.text((legend_x + 40, legend_y + 65), "RKNPU Usage (%)", fill='black', font=font)
        
        draw.line([legend_x, legend_y + 90, legend_x + 30, legend_y + 90], fill='purple', width=3)
        draw.text((legend_x + 40, legend_y + 85), "CPU Usage (%)", fill='black', font=font)

        draw.line([legend_x, legend_y + 110, legend_x + 30, legend_y + 110], fill='red', width=3)
        draw.text((legend_x + 40, legend_y + 105), "GPU Usage (%)", fill='black', font=font)

        draw.line([legend_x, legend_y + 130, legend_x + 30, legend_y + 130], fill='teal', width=3)
        draw.text((legend_x + 40, legend_y + 125), "Hailo Occupancy (%)", fill='black', font=font)

        # Dashed lines (shifted down to make room for the Hailo Usage entry above)
        draw_dashed_line(draw, (legend_x, legend_y + 150), (legend_x + 30, legend_y + 150), fill='blue')
        draw.text((legend_x + 40, legend_y + 145), "Inference Time Average", fill='black', font=font)

        draw_dashed_line(draw, (legend_x, legend_y + 170), (legend_x + 30, legend_y + 170), fill='green')
        draw.text((legend_x + 40, legend_y + 165), "FPS Average", fill='black', font=font)

        draw_dashed_line(draw, (legend_x, legend_y + 190), (legend_x + 30, legend_y + 190), fill='darkorange')
        draw.text((legend_x + 40, legend_y + 185), "RKNPU Average", fill='black', font=font)

        draw_dashed_line(draw, (legend_x, legend_y + 210), (legend_x + 30, legend_y + 210), fill='darkviolet')
        draw.text((legend_x + 40, legend_y + 205), "CPU Average", fill='black', font=font)

        draw_dashed_line(draw, (legend_x, legend_y + 230), (legend_x + 30, legend_y + 230), fill='darkred')
        draw.text((legend_x + 40, legend_y + 225), "GPU Average", fill='black', font=font)
        
        # Save image with higher DPI for better readability (keep pixel sizes unchanged)
        try:
            img.save(output_path, 'PNG', dpi=(300, 300))
        except TypeError:
            # PIL versions older may not accept dpi param for PNG; fallback
            img.save(output_path, 'PNG')
        return output_path
        
    except Exception as e:
        print(f"Error generating performance reports: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_csv_analysis(csv_filepath):
    """Print detailed analysis of CSV performance data"""
    try:
        import csv
        from statistics import mean
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data:
            print("No data found in CSV file")
            return
        
        # Helper to extract numeric data
        def get_col(col_name):
            return [float(row[col_name]) for row in data if col_name in row and row[col_name]]
            
        inference_times = get_col('inference_time_ms')
        fps_data = get_col('fps_actual')
        cpu_usage = get_col('cpu_usage_percent')
        npu_core0 = get_col('npu_core0_percent')
        npu_core1 = get_col('npu_core1_percent')
        npu_core2 = get_col('npu_core2_percent')
        gpu_usage = get_col('gpu_usage_percent')
        hailo_usage = get_col('hailo_usage_percent')
        rk3588_temp = get_col('rk3588_temp_c')
        hailo_temp = get_col('hailo_temp_c')
        detections = get_col('detections_count')
        
        detection_frames = sum(1 for d in detections if d > 0)
        
        print("\n" + "="*60)
        print("PERFORMANCE DATA ANALYSIS")
        print("="*60)
        
        def safe_mean(arr): return mean(arr) if arr else 0.0
        def safe_max(arr): return max(arr) if arr else 0.0
        
        # Explicit output requested by user
        print(f"Total frames analizados []: {len(data)}")
        print(f"Frames with Detections [count]: {detection_frames}")
        print(f"Inference Time Max [ms]: {safe_max(inference_times):.2f}")
        print(f"Inference Time Average [ms]: {safe_mean(inference_times):.2f}")
        print(f"FPS Avg [FPS]: {safe_mean(fps_data):.2f}")
        print(f"CPU Load Avg [%]: {safe_mean(cpu_usage):.1f}")
        print(f"GPU Avg [%]: {safe_mean(gpu_usage):.1f}")
        print(f"RKNPU Core0 Avg [%]: {safe_mean(npu_core0):.1f}")
        print(f"RKNPU Core1 Avg [%]: {safe_mean(npu_core1):.1f}")
        print(f"RKNPU Core2 Avg [%]: {safe_mean(npu_core2):.1f}")
        print(f"Hailo-8 Load Avg [%]: {safe_mean(hailo_usage):.1f}")
        print(f"RK3588 Max Temp [°C]: {safe_max(rk3588_temp):.1f}")
        print(f"Hailo-8 Temp Max [°C]: {safe_max(hailo_temp):.1f}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error analyzing CSV: {e}")

def find_latest_csv(device_name="NPU"):
    """Find the most recent performance CSV file"""
    search_patterns = [
        f"performance_metrics_{device_name}_*.csv",
        f"src/processing/results/performance_metrics_{device_name}_*.csv"
    ]
    
    csv_files = []
    for pattern in search_patterns:
        csv_files.extend(glob.glob(pattern))
    
    if not csv_files:
        return None
    
    return max(csv_files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description="Analyze performance metrics CSV files")
    parser.add_argument("csv_file", nargs="?", help="Path to CSV file to analyze")
    parser.add_argument("--latest", action="store_true", help="Analyze the latest CSV file")
    parser.add_argument("--device", default="NPU", help="Device type for latest file search")
    parser.add_argument("--export", help="Export analysis to text file")
    
    args = parser.parse_args()
    
    # Determine which CSV file to analyze
    if args.latest:
        csv_file = find_latest_csv(args.device)
        if not csv_file:
            print(f"No CSV files found for device: {args.device}")
            return
        print(f"Analyzing latest file: {csv_file}")
    elif args.csv_file:
        csv_file = args.csv_file
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            return
    else:
        parser.print_help()
        return
    
    print(f"\\nAnalyzing: {csv_file}")
    
    # Export analysis if requested
    if args.export:
        try:
            import io
            import contextlib
            
            # Capture print output
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                print_csv_analysis(csv_file)
            
            # Save to file
            with open(args.export, 'w', encoding='utf-8') as f:
                f.write(f"Performance Analysis Report\\n")
                f.write(f"Source: {csv_file}\\n")
                f.write(f"Generated: {time.ctime()}\\n\\n")
                f.write(output_buffer.getvalue())
            
            print(f"Analysis exported to: {args.export}")
        except Exception as e:
            print(f"Error exporting analysis: {e}")
    else:
        # Print to console
        print_csv_analysis(csv_file)
    
    # Generate performance reports
    print("\\nGenerating performance reports...")
    try:
        base_name = os.path.splitext(csv_file)[0]
        png_path = f"{base_name}_report.png"
        
        generated_report = generate_performance_reports(csv_file, png_path)
        if generated_report:
            print(f"Performance reports saved to: {generated_report}")
        else:
            print("Failed to generate performance reports")
    except Exception as e:
        print(f"Error generating reports: {e}")

if __name__ == "__main__":
    main()