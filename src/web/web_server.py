# -*- coding: utf-8 -*-
"""web_server.py
Flask web server for YOLO RKNN web interface
by fcascan 2025
"""
import os
import sys
import time
import queue
import threading
import json
import configparser
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import logging
from io import StringIO
import base64

# Import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.config import *

class WebServer:
    """Web server for YOLO RKNN interface"""
    
    def __init__(self, host='0.0.0.0', port=8080, http_logging=False):
        self.host = host
        self.port = port
        self.http_logging = http_logging
        self.app = Flask(__name__, 
                        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                        static_folder=os.path.join(os.path.dirname(__file__), 'static'))
        self.app.config['SECRET_KEY'] = 'yolo_rknn_secret_key'
        # Configure SocketIO with better compatibility
        self.socketio = SocketIO(self.app, 
                                cors_allowed_origins="*",
                                async_mode='threading',
                                transports=['websocket', 'polling'],
                                logger=False,
                                engineio_logger=False)
        
        # Video streaming integration
        from .video_integration import get_video_stream_manager
        self.video_manager = get_video_stream_manager()
        
        # Console logging integration
        from .console_integration import get_console_capture
        self.console_capture = get_console_capture()
        
        # Processing control
        self.processing_active = False
        self.processing_thread = None
        
        # Setup routes
        self.setup_routes()
        self.setup_socketio()
        

            
    def generate_video_stream(self, camera_id=None):
        """Generate video stream for web interface"""
        while True:
            frame = self.video_manager.get_latest_frame(camera_id)
            if frame is not None:
                # Encode frame to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_data = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                else:
                    # Send a blank frame if encoding fails
                    blank_frame = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9'
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
            else:
                # Send a placeholder message frame
                try:
                    import numpy as np
                    import cv2
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, 'No video stream available', (120, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(placeholder, 'Start processing to see live video', (100, 260), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                    ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        frame_data = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                except Exception as e:
                    # Fallback minimal frame
                    pass
            time.sleep(0.05)  # 20 FPS for better stability
            
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main web interface"""
            return render_template('index.html')
            
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route (main camera)"""
            try:
                return Response(self.generate_video_stream(),
                              mimetype='multipart/x-mixed-replace; boundary=frame',
                              headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                                     'Pragma': 'no-cache',
                                     'Expires': '0'})
            except Exception as e:
                return "Video feed error", 500
                
        @self.app.route('/video_feed/<int:camera_id>')
        def video_feed_camera(camera_id):
            """Video streaming route for specific camera"""
            try:
                return Response(self.generate_video_stream(camera_id),
                              mimetype='multipart/x-mixed-replace; boundary=frame',
                              headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                                     'Pragma': 'no-cache',
                                     'Expires': '0'})
            except Exception as e:
                return f"Video feed error for camera {camera_id}", 500
                          
        @self.app.route('/api/config', methods=['GET'])
        def get_config():
            """Get current configuration"""
            # Import fresh values to ensure we get the latest configuration
            from ..core.config import BENCHMARK_MODE, INFERENCE_DEVICE, MODEL_PATH, ONNX_MODEL_PATH
            from ..core.config import VIDEO_FILE_PATH, IMG_SIZE, FPS_TEXT_SIZE, LABEL_TEXT_SIZE
            from ..core.config import MAX_CAMERAS_TO_SCAN, CLASSES
            
            config_data = {
                'benchmark_mode': BENCHMARK_MODE,
                'inference_device': INFERENCE_DEVICE,
                'paths': {
                    'model_rknn': MODEL_PATH,
                    'model_onnx': ONNX_MODEL_PATH,
                    'video_file': VIDEO_FILE_PATH
                },
                'image_config': {
                    'img_size': IMG_SIZE,
                    'fps_text_size': FPS_TEXT_SIZE,
                    'label_text_size': LABEL_TEXT_SIZE
                },
                'camera_config': {
                    'max_cameras': MAX_CAMERAS_TO_SCAN
                },
                'classes': CLASSES
            }
            return jsonify(config_data)
            
        @self.app.route('/api/config', methods=['POST'])
        def update_config():
            """Update configuration"""
            try:
                data = request.get_json()
                
                # Update config.ini file
                config_path = os.path.join(BASE_DIR, 'config.ini')
                parser = configparser.ConfigParser()
                parser.read(config_path)
                
                if 'benchmark_mode' in data:
                    parser.set('MODE', 'benchmark_mode', str(data['benchmark_mode']).lower())
                    
                if 'inference_device' in data:
                    parser.set('INFERENCE', 'device', data['inference_device'])
                    
                if 'max_cameras' in data:
                    parser.set('CAMERA', 'max_cameras_to_scan', str(data['max_cameras']))
                
                # Write updated config
                with open(config_path, 'w') as configfile:
                    parser.write(configfile)
                
                # Reload configuration to make changes effective immediately
                from ..core.config import reload_config
                updated_config = reload_config()
                
                # Notify about changes via SocketIO
                self.socketio.emit('config_updated', {
                    'message': 'Configuration updated and reloaded',
                    'config': updated_config
                })
                    
                return jsonify({'status': 'success', 'message': 'Configuration updated and reloaded successfully'})
                
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/api/start_processing', methods=['POST'])
        def start_processing():
            """Start YOLO processing"""
            if self.processing_active:
                return jsonify({'status': 'error', 'message': 'Processing already active'}), 400
                
            try:
                # Start processing in separate thread
                self.processing_thread = threading.Thread(target=self._run_processing, daemon=True)
                self.processing_active = True
                self.processing_thread.start()
                
                return jsonify({'status': 'success', 'message': 'Processing started'})
                
            except Exception as e:
                self.processing_active = False
                return jsonify({'status': 'error', 'message': str(e)}), 500
                
        @self.app.route('/api/stop_processing', methods=['POST'])
        def stop_processing():
            """Stop YOLO processing"""
            if not self.processing_active:
                return jsonify({'status': 'error', 'message': 'No processing active'}), 400
                
            self.processing_active = False
            return jsonify({'status': 'success', 'message': 'Processing stopped'})
            
        @self.app.route('/api/status')
        def get_status():
            """Get current system status"""
            # Import fresh values to ensure we get the latest configuration
            from ..core.config import BENCHMARK_MODE, INFERENCE_DEVICE
            
            status = {
                'processing_active': self.processing_active,
                'current_mode': 'benchmark' if BENCHMARK_MODE else 'camera',
                'inference_device': INFERENCE_DEVICE,
                'frame_available': self.video_manager.get_latest_frame() is not None
            }
            return jsonify(status)
            
        @self.app.route('/api/system_monitor')
        def get_system_monitor():
            """Get system monitoring data"""
            try:
                monitor_data = self._get_system_monitoring_data()
                return jsonify(monitor_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/cameras')
        def get_cameras():
            """Get information about available cameras"""
            try:
                camera_count = self.video_manager.get_camera_count()
                return jsonify({
                    'camera_count': camera_count,
                    'cameras': [{'id': i, 'name': f'Camera {i}'} for i in range(camera_count)]
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/download/latest_csv')
        def download_latest_csv():
            """Download the latest CSV performance file"""
            import glob
            from flask import send_file
            
            try:
                # Search for latest CSV files
                csv_pattern = os.path.join(BASE_DIR, 'src', 'processing', 'results', 'performance_metrics_*.csv')
                csv_files = glob.glob(csv_pattern)
                
                if not csv_files:
                    return jsonify({'error': 'No CSV files found'}), 404
                
                # Get the most recent file
                latest_csv = max(csv_files, key=os.path.getmtime)
                
                return send_file(latest_csv, 
                               as_attachment=True,
                               download_name=os.path.basename(latest_csv),
                               mimetype='text/csv')
                               
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/download/latest_graphs')
        def download_latest_graphs():
            """Download the latest PNG graphs file"""
            import glob
            from flask import send_file
            
            try:
                # Search for latest PNG files
                png_pattern = os.path.join(BASE_DIR, 'src', 'processing', 'results', 'performance_metrics_*_graphs.png')
                png_files = glob.glob(png_pattern)
                
                if not png_files:
                    return jsonify({'error': 'No graph files found'}), 404
                
                # Get the most recent file
                latest_png = max(png_files, key=os.path.getmtime)
                
                return send_file(latest_png,
                               as_attachment=True, 
                               download_name=os.path.basename(latest_png),
                               mimetype='image/png')
                               
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
    def setup_socketio(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            # Send recent console messages
            messages = self.console_capture.get_messages(50)
            if messages:
                emit('console_messages', messages)
                
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            pass
            
        @self.socketio.on('request_console_update')
        def handle_console_request():
            """Send console updates to client"""
            messages = self.console_capture.get_messages(10)
            if messages:
                emit('console_messages', messages)
                
    def _get_system_monitoring_data(self):
        """Get real-time system monitoring data"""
        import psutil
        from ..utils.my_htop import get_npu_info, get_gpu_info, get_cpu_info
        
        monitor_data = {}
        
        # CPU Information
        try:
            cpu_loads, cpu_freqs = get_cpu_info()
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Convert dictionaries to ordered arrays
            cpu_loads_array = [cpu_loads.get(i, 0) for i in sorted(cpu_loads.keys())]
            cpu_freqs_array = [cpu_freqs.get(i, 0) for i in sorted(cpu_freqs.keys())]
            
            monitor_data['cpu'] = {
                'loads': cpu_loads_array,
                'frequencies': cpu_freqs_array,
                'average_load': sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0,
                'core_count': len(cpu_percent)
            }
        except Exception as e:
            monitor_data['cpu'] = {'error': str(e)}
            
        # NPU Information  
        try:
            npu_loads, npu_freq = get_npu_info()
            monitor_data['npu'] = {
                'loads': npu_loads,
                'frequency': npu_freq,
                'average_load': sum(npu_loads) / len(npu_loads) if npu_loads else 0,
                'core_count': len(npu_loads)
            }
        except Exception as e:
            monitor_data['npu'] = {'error': str(e)}
            
        # GPU Information
        try:
            gpu_load, gpu_freq = get_gpu_info()
            monitor_data['gpu'] = {
                'load': gpu_load if gpu_load is not None else 0,
                'frequency': gpu_freq if gpu_freq is not None else 0,
                'available': gpu_load is not None
            }
        except Exception as e:
            monitor_data['gpu'] = {'error': str(e)}
            
        # Memory Information
        try:
            memory = psutil.virtual_memory()
            monitor_data['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percentage': memory.percent,
                'total_gb': round(memory.total / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2)
            }
        except Exception as e:
            monitor_data['memory'] = {'error': str(e)}
            
        # Temperature Information
        try:
            # Try to get temperature from common Orange Pi thermal zones
            temp_paths = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp',
                '/sys/class/thermal/thermal_zone2/temp'
            ]
            temperatures = {}
            for i, temp_path in enumerate(temp_paths):
                try:
                    with open(temp_path, 'r') as f:
                        temp_raw = int(f.read().strip())
                        temp_celsius = temp_raw / 1000.0  # Convert from millidegrees
                        temperatures[f'zone_{i}'] = temp_celsius
                except:
                    continue
                    
            if temperatures:
                avg_temp = sum(temperatures.values()) / len(temperatures)
                monitor_data['temperature'] = {
                    'zones': temperatures,
                    'average': round(avg_temp, 1),
                    'max': round(max(temperatures.values()), 1)
                }
            else:
                monitor_data['temperature'] = {'error': 'No temperature sensors available'}
        except Exception as e:
            monitor_data['temperature'] = {'error': str(e)}
            
        return monitor_data
        
    def _run_processing(self):
        """Run YOLO processing with web integration"""
        try:
            from src.core.system_setup import setup_system, setup_inference_device, disable_unnecessary_logging
            from src.processing.yolo_post import yolo_onnx_postprocess
            from .web_video_processing import process_video_web
            from .web_camera_processing import process_cameras_web
            from .console_integration import start_console_capture, get_web_logger
            
            # Start console capture for web display
            start_console_capture()
            logger = get_web_logger()
            
            # Setup system dependencies and permissions
            setup_system()
            
            # Import fresh configuration values
            from ..core.config import BENCHMARK_MODE, INFERENCE_DEVICE
            
            # Setup and configure inference device with current config
            actual_device, rknn_available, rknn_modules = setup_inference_device(INFERENCE_DEVICE)
            
            # Disable unnecessary logging
            disable_unnecessary_logging()
            
            logger.info(f"Starting processing in {'BENCHMARK' if BENCHMARK_MODE else 'CAMERA'} mode using {actual_device}")
            
            if BENCHMARK_MODE:
                process_video_web(yolo_onnx_postprocess, self)
            else:
                process_cameras_web(yolo_onnx_postprocess, self)
                
        except Exception as e:
            logger.error(f"Processing failed: {e}")
        finally:
            self.processing_active = False

        
    def start_console_broadcaster(self):
        """Start console message broadcaster"""
        def broadcast_console():
            while True:
                try:
                    messages = self.console_capture.get_messages(10)
                    if messages:
                        self.socketio.emit('console_messages', messages)
                        
                except Exception as e:
                    pass
                    
                time.sleep(0.5)  # Update every 500ms
                
        broadcaster_thread = threading.Thread(target=broadcast_console, daemon=True)
        broadcaster_thread.start()
        
    def run(self, debug=False):
        """Start the web server"""
        import subprocess
        import socket
        
        # Get local IP address with multiple methods
        local_ip = '127.0.0.1'
        try:
            # Method 1: Get IP from route
            result = subprocess.run(['ip', 'route', 'get', '1.1.1.1'], capture_output=True, text=True)
            if result.returncode == 0 and len(result.stdout.split()) > 6:
                local_ip = result.stdout.split()[6]
        except:
            # Fallback method
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
            except:
                local_ip = '127.0.0.1'
            finally:
                s.close()
        
        print("="*50)
        print(f"[INFO] Web Interface: http://{local_ip}:{self.port}")
        print(f"[INFO] Device IP: {local_ip}")
        print("="*50)
        
        try:
            self.start_console_broadcaster()
            
            # Configure HTTP request logging
            if not self.http_logging:
                # Disable Werkzeug request logging
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                
            self.socketio.run(self.app, 
                            host=self.host, 
                            port=self.port, 
                            debug=debug, 
                            allow_unsafe_werkzeug=True,
                            use_reloader=False)
        except Exception as e:
            print(f"[ERROR] Server error: {e}")

def create_web_server(host='0.0.0.0', port=8080, http_logging=False):
    """Factory function to create web server instance"""
    return WebServer(host, port, http_logging)