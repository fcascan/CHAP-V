import os
import sys
import argparse
import configparser
import subprocess
import logging

def setup_logger(log_file="run_all_benchmarks.log"):
    logger = logging.getLogger("BenchmarkRunner")
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def update_config(config_path, params, logger):
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
        
    config.read(config_path)
    
    try:
        # Update INFERENCE params
        if 'INFERENCE' not in config:
            config['INFERENCE'] = {}
        config['INFERENCE']['inference_device'] = params['mode']
        if params['instances'] is not None:
            config['INFERENCE']['max_inference_instances'] = str(params['instances'])
        if params['timeout'] is not None:
            config['INFERENCE']['inference_timeout_minutes'] = str(params['timeout'])
        if params['ignore_frames'] is not None:
            config['INFERENCE']['ignore_initial_frames'] = str(params['ignore_frames'])
        if params['ignore_final_frames'] is not None:
            config['INFERENCE']['ignore_final_frames'] = str(params['ignore_final_frames'])
        if params['graph_method'] is not None:
            config['INFERENCE']['graph_downsample_method'] = params['graph_method']
            
        # Update MODE params
        if 'MODE' not in config:
            config['MODE'] = {}
        if params['benchmark_mode'] is not None:
            config['MODE']['benchmark_mode'] = str(params['benchmark_mode']).lower()
        if params['benchmark_loop'] is not None:
            config['MODE']['benchmark_loop'] = str(params['benchmark_loop']).lower()
            
        # Update PATHS params
        if 'PATHS' not in config:
            config['PATHS'] = {}
        
        size = params['size']
        size_l = size.lower()
        config['PATHS']['model_rknn'] = f"assets/models/threats_{size}/threats_{size_l}.rknn"
        config['PATHS']['model_onnx'] = f"assets/models/threats_{size}/threats_{size_l}.onnx"
        config['PATHS']['model_mnn'] = f"assets/models/threats_{size}/threats_{size_l}.mnn"
        config['PATHS']['model_hailo8'] = f"assets/models/threats_{size}/threats_{size_l}.hef"
        
        with open(config_path, 'w') as configfile:
            config.write(configfile)
            
        return True
    except Exception as e:
        logger.error(f"Error updating config.ini: {e}")
        return False

def run_main(logger):
    # Run main.py and stream output line by line to logger
        
    process = subprocess.Popen(
        [sys.executable, 'main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in iter(process.stdout.readline, ''):
        line = line.rstrip()
        if line:
            logger.info(f"[main.py] {line}")
            
    process.stdout.close()
    return_code = process.wait()
    return return_code

def main():
    parser = argparse.ArgumentParser(
        description="Automate benchmarking across multiple models and inference modes for CHAP-V.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default settings
  python run_all_benchmarks.py

  # Run with custom instances and timeout
  python run_all_benchmarks.py --instances 1 --timeout 5

  # Run only 1 iteration of benchmark_loop
  python run_all_benchmarks.py --no-loop
"""
    )
    
    parser.add_argument('--benchmark_mode', dest='benchmark_mode', action='store_true', default=None,
                        help="Enable benchmark mode (video files). Defaults to True if omitted.")
    parser.add_argument('--no-benchmark_mode', dest='benchmark_mode', action='store_false',
                        help="Disable benchmark mode (live cameras).")
    
    parser.add_argument('--benchmark_loop', dest='benchmark_loop', action='store_true', default=None,
                        help="Enable continuous loop in benchmark mode. Defaults to True if omitted.")
    parser.add_argument('--no-loop', dest='benchmark_loop', action='store_false',
                        help="Disable continuous loop (stops after video ends).")
                        
    parser.add_argument('--instances', type=int, default=None,
                        help="Number of parallel inference streams (max_inference_instances).")
                        
    parser.add_argument('--timeout', type=int, default=None,
                        help='Inference timeout in minutes')
    parser.add_argument('--ignore_frames', type=int, default=None,
                        help='Number of initial frames to ignore in analysis')
    parser.add_argument('--ignore_final_frames', type=int, default=None,
                        help='Number of final frames to ignore in analysis')
    parser.add_argument('--graph_method', type=str, choices=['worst_case', 'mean'], default=None,
                        help='Graph downsampling method (worst_case or mean)')
    
    args = parser.parse_args()
    
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Starting Automated Benchmark Suite")
    logger.info("=" * 60)
    
    models = ['N', 'S', 'M', 'L', 'X']
    modes = [
        'RKNPU-Auto',
        'RKNPU-Distributed',
        'CPU',
        'CPU-50%',
        'GPU-OpenCV-OpenCL',
        'GPU-MNN',
        'NPU-Hailo8'
    ]
    
    config_path = 'config.ini'
    total_combinations = len(models) * len(modes)
    current_iteration = 0
    
    for model_size in models:
        for mode in modes:
            current_iteration += 1
            logger.info("-" * 60)
            logger.info(f"Running iteration {current_iteration}/{total_combinations} -> Model: {model_size} | Mode: {mode}")
            
            # Prepare configuration params
            params = {
                'size': model_size,
                'mode': mode,
                'benchmark_mode': args.benchmark_mode if args.benchmark_mode is not None else True,
                'benchmark_loop': args.benchmark_loop if args.benchmark_loop is not None else True,
                'instances': args.instances,
                'timeout': args.timeout,
                'ignore_frames': args.ignore_frames,
                'ignore_final_frames': args.ignore_final_frames,
                'graph_method': args.graph_method
            }
            
            if not update_config(config_path, params, logger):
                logger.error(f"Failed to setup config for {model_size} | {mode}. Skipping.")
                continue
                
            logger.info(f"Configuration successfully applied for {model_size} | {mode}. Launching main.py...")
            
            try:
                ret_code = run_main(logger)
                if ret_code == 0:
                    logger.info(f"Iteration completed successfully (Return Code: {ret_code})")
                else:
                    logger.warning(f"Iteration completed with non-zero return code: {ret_code}")
            except Exception as e:
                logger.error(f"Exception while running main.py for {model_size} | {mode}: {e}")
                
            logger.info("-" * 60)
            
    logger.info("=" * 60)
    logger.info("Automated Benchmark Suite Completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
