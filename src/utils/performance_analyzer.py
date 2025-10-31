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

def generate_performance_graphs(csv_filepath, output_path=None):
    """
    Generate performance analysis graphs as PNG file using PIL.
    
    Args:
        csv_filepath (str): Path to the CSV file
        output_path (str): Output path for the PNG file (optional)
    
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
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
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
        
        if not data or len(data.get('frame_number', [])) == 0:
            print(f"No valid data found in {csv_filepath}")
            return None

        # Create image
        img_width, img_height = 1400, 1000
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
        draw.text((20, 50), subtitle, fill='gray', font=font)
        draw.text((20, 75), f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fill='gray', font=small_font)
        
        # Statistics summary
        y_pos = 110
        stats_text = [
            f"SUMMARY STATISTICS",
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
                draw.text((20, y_pos), text, fill='navy', font=font)
            else:
                draw.text((20, y_pos), text, fill='black', font=small_font)
            y_pos += 25
        
        # Graph 1: Inference Time Timeline
        graph_y = 380
        graph_width = 650
        graph_height = 180
        
        # Draw inference time graph
        draw.rectangle([20, graph_y, 20 + graph_width, graph_y + graph_height], outline='black', width=2)
        draw.text((20, graph_y - 25), "Inference Time Over Time (ms)", fill='black', font=font)
        
        if len(inference_times) > 0:
            max_val = max(inference_times)
            min_val = min(inference_times)
            val_range = max_val - min_val if max_val != min_val else 1
            
            # Sample data points if too many
            sample_size = min(300, len(inference_times))
            step = len(inference_times) // sample_size if sample_size > 0 else 1
            sampled_data = inference_times[::step][:sample_size]
            
            # Draw grid lines
            for i in range(5):
                y_grid = graph_y + (i * graph_height // 4)
                draw.line([20, y_grid, 20 + graph_width, y_grid], fill='lightgray', width=1)
                val = max_val - (i * val_range / 4)
                draw.text((25, y_grid - 8), f"{val:.1f}", fill='gray', font=small_font)
            
            if len(sampled_data) > 1:
                for i in range(len(sampled_data) - 1):
                    x1 = 20 + (i * graph_width // len(sampled_data))
                    y1 = graph_y + graph_height - int((sampled_data[i] - min_val) / val_range * graph_height)
                    x2 = 20 + ((i + 1) * graph_width // len(sampled_data))
                    y2 = graph_y + graph_height - int((sampled_data[i + 1] - min_val) / val_range * graph_height)
                    draw.line([x1, y1, x2, y2], fill='blue', width=2)
                
                # Draw mean line
                mean_y = graph_y + graph_height - int((inf_mean - min_val) / val_range * graph_height)
                draw.line([20, mean_y, 20 + graph_width, mean_y], fill='red', width=2)
                draw.text((25, mean_y - 15), f"Mean: {inf_mean:.1f}ms", fill='red', font=small_font)
        
        # Graph 2: FPS Timeline
        graph2_x = 720
        draw.rectangle([graph2_x, graph_y, graph2_x + graph_width, graph_y + graph_height], outline='black', width=2)
        draw.text((graph2_x, graph_y - 25), "FPS Over Time", fill='black', font=font)
        
        if len(fps_data) > 0:
            max_fps = max(fps_data)
            min_fps = min(fps_data) 
            fps_range = max_fps - min_fps if max_fps != min_fps else 1
            
            sampled_fps = fps_data[::step][:sample_size]
            
            # Draw grid lines
            for i in range(5):
                y_grid = graph_y + (i * graph_height // 4)
                draw.line([graph2_x, y_grid, graph2_x + graph_width, y_grid], fill='lightgray', width=1)
                val = max_fps - (i * fps_range / 4)
                draw.text((graph2_x + 5, y_grid - 8), f"{val:.1f}", fill='gray', font=small_font)
            
            if len(sampled_fps) > 1:
                for i in range(len(sampled_fps) - 1):
                    x1 = graph2_x + (i * graph_width // len(sampled_fps))
                    y1 = graph_y + graph_height - int((sampled_fps[i] - min_fps) / fps_range * graph_height)
                    x2 = graph2_x + ((i + 1) * graph_width // len(sampled_fps))
                    y2 = graph_y + graph_height - int((sampled_fps[i + 1] - min_fps) / fps_range * graph_height)
                    draw.line([x1, y1, x2, y2], fill='green', width=2)
                
                # Draw mean line
                mean_y = graph_y + graph_height - int((fps_mean - min_fps) / fps_range * graph_height)
                draw.line([graph2_x, mean_y, graph2_x + graph_width, mean_y], fill='red', width=2)
                draw.text((graph2_x + 5, mean_y - 15), f"Mean: {fps_mean:.1f} FPS", fill='red', font=small_font)
        
        # Graph 3: NPU vs CPU Usage
        graph3_y = 600
        graph3_height = 150
        
        draw.rectangle([20, graph3_y, 20 + graph_width, graph3_y + graph3_height], outline='black', width=2)
        draw.text((20, graph3_y - 25), "NPU vs CPU Usage (%)", fill='black', font=font)
        
        if len(npu_core0) > 0 and len(cpu_usage) > 0:
            sampled_npu = npu_core0[::step][:sample_size]
            sampled_cpu = cpu_usage[::step][:sample_size]
            
            # Draw grid lines
            for i in range(5):
                y_grid = graph3_y + (i * graph3_height // 4)
                draw.line([20, y_grid, 20 + graph_width, y_grid], fill='lightgray', width=1)
                val = 100 - (i * 25)
                draw.text((25, y_grid - 8), f"{val}%", fill='gray', font=small_font)
            
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
        
        # Legend
        legend_x = 720
        legend_y = 600
        draw.text((legend_x, legend_y), "LEGEND:", fill='black', font=font)
        draw.line([legend_x, legend_y + 30, legend_x + 30, legend_y + 30], fill='blue', width=3)
        draw.text((legend_x + 40, legend_y + 25), "Inference Time", fill='black', font=small_font)
        draw.line([legend_x, legend_y + 50, legend_x + 30, legend_y + 50], fill='green', width=3)
        draw.text((legend_x + 40, legend_y + 45), "FPS", fill='black', font=small_font)
        draw.line([legend_x, legend_y + 70, legend_x + 30, legend_y + 70], fill='orange', width=3)
        draw.text((legend_x + 40, legend_y + 65), "NPU Usage", fill='black', font=small_font)
        draw.line([legend_x, legend_y + 90, legend_x + 30, legend_y + 90], fill='purple', width=3)
        draw.text((legend_x + 40, legend_y + 85), "CPU Usage", fill='black', font=small_font)
        draw.line([legend_x, legend_y + 110, legend_x + 30, legend_y + 110], fill='red', width=3)
        draw.text((legend_x + 40, legend_y + 105), "Mean Values", fill='black', font=small_font)
        
        # Save image
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