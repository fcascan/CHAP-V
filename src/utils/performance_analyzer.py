#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""performance_analyzer.py
Performance analysis and visualization module for YOLO RKNN processing metrics
by fcascan 2025
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

def generate_performance_graphs(csv_filepath, output_path=None, npu_core_id=None, model_name=None,
                                benchmark_video=None, camera_index=None, inference_device=None):
    """
    Generate performance analysis graphs as PNG file using PIL.

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
        print("PIL not available for graph generation")
        return None
        
    try:
        # Determine output path
        if output_path is None:
            base_name = os.path.splitext(csv_filepath)[0]
            output_path = f"{base_name}_graphs.png"
        
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
        
        # Calculate statistics
        inf_mean = mean(inference_times)
        inf_median = median(inference_times)
        inf_95th = percentile(inference_times, 95)
        fps_mean = mean(fps_data)
        npu_mean = mean(npu_core0)
        cpu_mean = mean(cpu_usage)
        
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
            draw.text((20, y_meta), f"NPU Core: {npu_core_id}", fill='black', font=font)
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
            f"  • Average inference time: {inf_mean:.1f} ms",
            f"  • Median inference time: {inf_median:.1f} ms", 
            f"  • 95th percentile: {inf_95th:.1f} ms",
            f"  • Average FPS: {fps_mean:.1f}",
            f"  • Average NPU usage: {npu_mean:.1f}%",
            f"  • Average CPU usage: {cpu_mean:.1f}%",
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

            sampled_count = len(sampled_data)
            sample_indices = [min(i * step, len(inference_times) - 1) for i in range(sampled_count)]
            # Show about 8 labels max, skip the first (zero) label
            label_interval = max(1, sampled_count // 8) if sampled_count > 0 else 1
            for idx_i in range(0, sampled_count, label_interval):
                # skip first label (zero)
                if idx_i == 0:
                    continue
                idx = sample_indices[idx_i]
                x_pos = 20 + (idx_i * graph_width // sampled_count)
                draw.line([x_pos, graph_y + graph_height, x_pos, graph_y + graph_height + 5], fill='black', width=1)
                # compute label from timestamps if available
                label = None
                if first_ts and idx < len(parsed_ts) and parsed_ts[idx]:
                    diff = (parsed_ts[idx] - first_ts).total_seconds()
                    if diff > 0:
                        if unit == 's':
                            label = f"{int(diff)}s"
                        elif unit == 'm':
                            label = f"{int(diff // 60)}m"
                        else:
                            label = f"{int(diff // 3600)}h"
                else:
                    # fallback to seconds using frame index (assuming ~30fps)
                    sec = idx / 30
                    if sec > 0:
                        label = f"{int(sec)}s"
                if label:
                    draw.text((x_pos - 15, graph_y + graph_height + 8), label, fill='gray', font=small_font)
            
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
            
            # Draw time axis labels using parsed timestamps when available (skip zero)
            sampled_count_fps = len(sampled_fps)
            sample_indices_fps = [min(i * step, len(fps_data) - 1) for i in range(sampled_count_fps)]
            label_interval_fps = max(1, sampled_count_fps // 8) if sampled_count_fps > 0 else 1
            for idx_i in range(0, sampled_count_fps, label_interval_fps):
                # skip first label (zero)
                if idx_i == 0:
                    continue
                idx = sample_indices_fps[idx_i]
                x_pos = graph2_x + (idx_i * graph_width // sampled_count_fps)
                draw.line([x_pos, graph_y + graph_height, x_pos, graph_y + graph_height + 5], fill='black', width=1)
                label = None
                if first_ts and idx < len(parsed_ts) and parsed_ts[idx]:
                    diff = (parsed_ts[idx] - first_ts).total_seconds()
                    if diff > 0:
                        if unit == 's':
                            label = f"{int(diff)}s"
                        elif unit == 'm':
                            label = f"{int(diff // 60)}m"
                        else:
                            label = f"{int(diff // 3600)}h"
                else:
                    sec = idx / 30
                    if sec > 0:
                        label = f"{int(sec)}s"
                if label:
                    draw.text((x_pos - 15, graph_y + graph_height + 8), label, fill='gray', font=small_font)
            
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
        _npu_title = f"NPU Core {npu_core_id} Usage (%)" if npu_core_id is not None else "NPU Usage (%)"
        draw.text((graph4_x, graph4_y - 25), _npu_title, fill='black', font=font_large)
        
        if len(npu_core0) > 0:
            sampled_npu = npu_core0[::step][:sample_size]
            
            # Draw grid lines
            for i in range(5):
                y_grid = graph4_y + (i * graph4_height // 4)
                draw.line([graph4_x, y_grid, graph4_x + graph4_width, y_grid], fill='lightgray', width=1)
                val = 100 - (i * 25)
                draw.text((graph4_x + 5, y_grid - 8), f"{val}%", fill='gray', font=small_font)
            
            # Draw time axis labels for NPU individual
            sampled_count_npu = len(sampled_npu)
            sample_indices_npu = [min(i * step, len(npu_core0) - 1) for i in range(sampled_count_npu)]
            label_interval_npu = max(1, sampled_count_npu // 8) if sampled_count_npu > 0 else 1
            for idx_i in range(0, sampled_count_npu, label_interval_npu):
                if idx_i == 0:
                    continue
                idx = sample_indices_npu[idx_i]
                x_pos = graph4_x + (idx_i * graph4_width // sampled_count_npu)
                draw.line([x_pos, graph4_y + graph4_height, x_pos, graph4_y + graph4_height + 5], fill='black', width=1)
                label = None
                if first_ts and idx < len(parsed_ts) and parsed_ts[idx]:
                    diff = (parsed_ts[idx] - first_ts).total_seconds()
                    if diff > 0:
                        if unit == 's':
                            label = f"{int(diff)}s"
                        elif unit == 'm':
                            label = f"{int(diff // 60)}m"
                        else:
                            label = f"{int(diff // 3600)}h"
                else:
                    sec = idx / 30
                    if sec > 0:
                        label = f"{int(sec)}s"
                if label:
                    draw.text((x_pos - 12, graph4_y + graph4_height + 8), label, fill='gray', font=small_font)
            
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
            
            # Draw time axis labels for CPU individual
            sampled_count_cpu = len(sampled_cpu)
            sample_indices_cpu = [min(i * step, len(cpu_usage) - 1) for i in range(sampled_count_cpu)]
            label_interval_cpu = max(1, sampled_count_cpu // 8) if sampled_count_cpu > 0 else 1
            for idx_i in range(0, sampled_count_cpu, label_interval_cpu):
                if idx_i == 0:
                    continue
                idx = sample_indices_cpu[idx_i]
                x_pos = graph5_x + (idx_i * graph5_width // sampled_count_cpu)
                draw.line([x_pos, graph5_y + graph5_height, x_pos, graph5_y + graph5_height + 5], fill='black', width=1)
                label = None
                if first_ts and idx < len(parsed_ts) and parsed_ts[idx]:
                    diff = (parsed_ts[idx] - first_ts).total_seconds()
                    if diff > 0:
                        if unit == 's':
                            label = f"{int(diff)}s"
                        elif unit == 'm':
                            label = f"{int(diff // 60)}m"
                        else:
                            label = f"{int(diff // 3600)}h"
                else:
                    sec = idx / 30
                    if sec > 0:
                        label = f"{int(sec)}s"
                if label:
                    draw.text((x_pos - 12, graph5_y + graph5_height + 8), label, fill='gray', font=small_font)
            
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
        
        # Graph 3: NPU vs CPU Usage with average lines (third row, full width)
        graph3_y = graph4_y + graph4_height + graph_row_sep
        graph3_height = 150
        
        draw.rectangle([20, graph3_y, 20 + graph_width, graph3_y + graph3_height], outline='black', width=2)
        _cmp_title = f"NPU Core {npu_core_id} vs CPU Usage Comparison (%)" if npu_core_id is not None else "NPU vs CPU Usage Comparison (%)"
        draw.text((20, graph3_y - 25), _cmp_title, fill='black', font=font_large)
        
        if len(npu_core0) > 0 and len(cpu_usage) > 0:
            sampled_npu = npu_core0[::step][:sample_size]
            sampled_cpu = cpu_usage[::step][:sample_size]
            
            # Draw grid lines
            for i in range(5):
                y_grid = graph3_y + (i * graph3_height // 4)
                draw.line([20, y_grid, 20 + graph_width, y_grid], fill='lightgray', width=1)
                val = 100 - (i * 25)
                draw.text((25, y_grid - 8), f"{val}%", fill='gray', font=small_font)
            
            # Draw time axis labels for comparison graph
            sampled_count_cmp = len(sampled_npu)
            sample_indices_cmp = [min(i * step, len(npu_core0) - 1) for i in range(sampled_count_cmp)]
            label_interval_cmp = max(1, sampled_count_cmp // 8) if sampled_count_cmp > 0 else 1
            for idx_i in range(0, sampled_count_cmp, label_interval_cmp):
                if idx_i == 0:
                    continue
                idx = sample_indices_cmp[idx_i]
                x_pos = 20 + (idx_i * graph_width // sampled_count_cmp)
                draw.line([x_pos, graph3_y + graph3_height, x_pos, graph3_y + graph3_height + 5], fill='black', width=1)
                label = None
                if first_ts and idx < len(parsed_ts) and parsed_ts[idx]:
                    diff = (parsed_ts[idx] - first_ts).total_seconds()
                    if diff > 0:
                        if unit == 's':
                            label = f"{int(diff)}s"
                        elif unit == 'm':
                            label = f"{int(diff // 60)}m"
                        else:
                            label = f"{int(diff // 3600)}h"
                else:
                    sec = idx / 30
                    if sec > 0:
                        label = f"{int(sec)}s"
                if label:
                    draw.text((x_pos - 15, graph3_y + graph3_height + 8), label, fill='gray', font=small_font)
            
            if len(sampled_npu) > 1:
                # NPU line
                for i in range(len(sampled_npu) - 1):
                    x1 = 20 + (i * graph_width // len(sampled_npu))
                    y1 = graph3_y + graph3_height - int(sampled_npu[i] * graph3_height / 100)
                    x2 = 20 + ((i + 1) * graph_width // len(sampled_npu))
                    y2 = graph3_y + graph3_height - int(sampled_npu[i + 1] * graph3_height / 100)
                    draw.line([x1, y1, x2, y2], fill='orange', width=2)
                
                # CPU line  
                for i in range(len(sampled_cpu) - 1):
                    x1 = 20 + (i * graph_width // len(sampled_cpu))
                    y1 = graph3_y + graph3_height - int(sampled_cpu[i] * graph3_height / 100)
                    x2 = 20 + ((i + 1) * graph_width // len(sampled_cpu))
                    y2 = graph3_y + graph3_height - int(sampled_cpu[i + 1] * graph3_height / 100)
                    draw.line([x1, y1, x2, y2], fill='purple', width=2)
                
                # Draw dashed average lines
                # npu_avg_y = graph3_y + graph3_height - int(npu_mean * graph3_height / 100)
                # cpu_avg_y = graph3_y + graph3_height - int(cpu_mean * graph3_height / 100)
                # draw_dashed_line(draw, (20, npu_avg_y), (20 + graph_width, npu_avg_y), fill='darkorange')
                # draw_dashed_line(draw, (20, cpu_avg_y), (20 + graph_width, cpu_avg_y), fill='darkviolet')
        
        # Add text labels below NPU vs CPU graph
        # draw.text((25, graph3_y + graph3_height + 20), f"NPU Avg: {npu_mean:.1f}%", fill='darkorange', font=font)
        # draw.text((25, graph3_y + graph3_height + 40), f"CPU Avg: {cpu_mean:.1f}%", fill='darkviolet', font=font)
        
        # Legend (updated position and content)
        legend_x = 20
        legend_y = graph3_y + graph3_height + 90
        draw.text((legend_x, legend_y), "LEGEND:", fill='black', font=font_large)
        
        # Solid lines
        draw.line([legend_x, legend_y + 30, legend_x + 30, legend_y + 30], fill='blue', width=3)
        draw.text((legend_x + 40, legend_y + 25), "Inference Time (ms)", fill='black', font=font)
        
        draw.line([legend_x, legend_y + 50, legend_x + 30, legend_y + 50], fill='green', width=3)
        draw.text((legend_x + 40, legend_y + 45), "FPS Over Time", fill='black', font=font)
        
        draw.line([legend_x, legend_y + 70, legend_x + 30, legend_y + 70], fill='orange', width=3)
        draw.text((legend_x + 40, legend_y + 65), "NPU Usage (%)", fill='black', font=font)
        
        draw.line([legend_x, legend_y + 90, legend_x + 30, legend_y + 90], fill='purple', width=3)
        draw.text((legend_x + 40, legend_y + 85), "CPU Usage (%)", fill='black', font=font)
        
        # Dashed lines
        draw_dashed_line(draw, (legend_x, legend_y + 110), (legend_x + 30, legend_y + 110), fill='blue')
        draw.text((legend_x + 40, legend_y + 105), "Inference Time Average", fill='black', font=font)

        draw_dashed_line(draw, (legend_x, legend_y + 130), (legend_x + 30, legend_y + 130), fill='green')
        draw.text((legend_x + 40, legend_y + 125), "FPS Average", fill='black', font=font)

        draw_dashed_line(draw, (legend_x, legend_y + 150), (legend_x + 30, legend_y + 150), fill='darkorange')
        draw.text((legend_x + 40, legend_y + 145), "NPU Average", fill='black', font=font)

        draw_dashed_line(draw, (legend_x, legend_y + 170), (legend_x + 30, legend_y + 170), fill='darkviolet')
        draw.text((legend_x + 40, legend_y + 165), "CPU Average", fill='black', font=font)
        
        # Save image with higher DPI for better readability (keep pixel sizes unchanged)
        try:
            img.save(output_path, 'PNG', dpi=(300, 300))
        except TypeError:
            # PIL versions older may not accept dpi param for PNG; fallback
            img.save(output_path, 'PNG')
        return output_path
        
    except Exception as e:
        print(f"Error generating performance graphs: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_csv_analysis(csv_filepath):
    """Print detailed analysis of CSV performance data"""
    try:
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data:
            print("No data found in CSV file")
            return
        
        # Extract numeric data
        inference_times = [float(row['inference_time_ms']) for row in data if row['inference_time_ms']]
        fps_data = [float(row['fps_actual']) for row in data if row['fps_actual']]
        cpu_usage = [float(row['cpu_usage_percent']) for row in data if row['cpu_usage_percent']]
        npu_core0 = [float(row['npu_core0_percent']) for row in data if row['npu_core0_percent']]
        npu_core1 = [float(row['npu_core1_percent']) for row in data if row['npu_core1_percent']]
        npu_core2 = [float(row['npu_core2_percent']) for row in data if row['npu_core2_percent']]
        gpu_usage = [float(row['gpu_usage_percent']) for row in data if row['gpu_usage_percent']]
        detections = [float(row['detections_count']) for row in data if row['detections_count']]
        
        print("\\n" + "="*60)
        print("PERFORMANCE DATA ANALYSIS")
        print("="*60)
        print(f"Total frames analyzed: {len(data)}")
        
        # Inference time analysis
        print(f"\\nINFERENCE TIME STATISTICS (ms)")
        print(f"  Mean: {mean(inference_times):.2f}")
        print(f"  Median: {median(inference_times):.2f}")
        print(f"  Std Dev: {std(inference_times):.2f}")
        print(f"  Min: {min(inference_times):.2f}")
        print(f"  Max: {max(inference_times):.2f}")
        print(f"  95th percentile: {percentile(inference_times, 95):.2f}")
        print(f"  99th percentile: {percentile(inference_times, 99):.2f}")
        
        # CPU usage analysis
        print(f"\\nCPU USAGE STATISTICS (%)")
        print(f"  Mean: {mean(cpu_usage):.1f}")
        print(f"  Median: {median(cpu_usage):.1f}")
        print(f"  Min: {min(cpu_usage):.1f}")
        print(f"  Max: {max(cpu_usage):.1f}")
        
        # NPU usage analysis
        print(f"\\nNPU USAGE STATISTICS (%)")
        
        for core_idx, core_data in enumerate([npu_core0, npu_core1, npu_core2]):
            active_samples = [x for x in core_data if x > 0]
            print(f"  NPU Core {core_idx}:")
            print(f"    Mean: {mean(core_data):.1f}")
            print(f"    Median: {median(core_data):.1f}")
            print(f"    Active samples: {len(active_samples)} ({len(active_samples)/len(core_data)*100:.1f}%)")
        
        # GPU usage analysis  
        active_gpu = [x for x in gpu_usage if x > 0]
        print(f"\\nGPU USAGE STATISTICS (%)")
        print(f"  Mean: {mean(gpu_usage):.1f}")
        print(f"  Median: {median(gpu_usage):.1f}")
        print(f"  Min: {min(gpu_usage):.1f}")
        print(f"  Max: {max(gpu_usage):.1f}")
        print(f"  Active samples: {len(active_gpu)} ({len(active_gpu)/len(gpu_usage)*100:.1f}%)")
        
        # FPS analysis
        print(f"\\nFPS STATISTICS")
        print(f"  Mean: {mean(fps_data):.2f}")
        print(f"  Median: {median(fps_data):.2f}")
        print(f"  Min: {min(fps_data):.2f}")
        print(f"  Max: {max(fps_data):.2f}")
        
        # Detection analysis
        detection_frames = sum(1 for d in detections if d > 0)
        print(f"\\nDETECTION STATISTICS")
        print(f"  Total detections: {sum(detections)}")
        print(f"  Mean detections per frame: {mean(detections):.2f}")
        print(f"  Frames with detections: {detection_frames}")
        print(f"  Detection rate: {detection_frames/len(detections)*100:.1f}%")
        
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
    
    # Generate performance graphs
    print("\\nGenerating performance graphs...")
    try:
        base_name = os.path.splitext(csv_file)[0]
        png_path = f"{base_name}_graphs.png"
        
        generated_graph = generate_performance_graphs(csv_file, png_path)
        if generated_graph:
            print(f"Performance graphs saved to: {generated_graph}")
        else:
            print("Failed to generate performance graphs")
    except Exception as e:
        print(f"Error generating graphs: {e}")

if __name__ == "__main__":
    main()