# =============================================================================
# Origin: https://github.com/airockchip/rknn_model_zoo
# Path:   examples/yolo11/python/convert.py
# License: Apache 2.0
# Copied and included in this repository to avoid runtime git-clone dependency.
# Modified from original:
#   - Switched from positional sys.argv parsing to argparse (named arguments)
#   - --dataset path is now a required CLI argument when quantizing
#   - --output path defaults to <model_stem>.rknn in the current directory
#   - --verbose flag controls RKNN logging verbosity
# =============================================================================

import os
import argparse
from rknn.api import RKNN


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert an ONNX model to RKNN format.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--model', '-m', required=True,
                        help='Path to the input ONNX model file.')
    parser.add_argument('--platform', '-p', required=True,
                        help='Target RKNPU platform.\n'
                             'Choices: rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808')
    parser.add_argument('--dtype', '-d', default='i8', choices=['i8', 'u8', 'fp'],
                        help="Quantization type:\n"
                             "  i8 / u8  – int8 quantization (default: i8)\n"
                             "  fp       – floating point (no quantization)")
    parser.add_argument('--output', '-o', default=None,
                        help='Output .rknn file path.\n'
                             'Default: <model_stem>.rknn in the current directory.')
    parser.add_argument('--dataset', default=None,
                        help='Path to calibration dataset .txt file.\n'
                             'Required when --dtype is i8 or u8.')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='Enable verbose RKNN logging.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    do_quant = args.dtype in ('i8', 'u8')

    # Validate dataset path when quantizing
    if do_quant:
        if not args.dataset:
            print('ERROR: --dataset is required when using int8/u8 quantization.')
            print('Pass --dtype fp to disable quantization, or provide a calibration dataset.')
            exit(1)
        if not os.path.isfile(args.dataset):
            print(f'ERROR: Dataset file not found: {args.dataset}')
            exit(1)

    # Derive default output path from model name
    if args.output:
        output_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.model))[0]
        output_path = os.path.join('.', stem + '.rknn')

    print(f'Model:    {args.model}')
    print(f'Platform: {args.platform}')
    print(f'Dtype:    {args.dtype} (quantize={do_quant})')
    print(f'Output:   {output_path}')
    if do_quant:
        print(f'Dataset:  {args.dataset}')

    # 1) Create RKNN object
    rknn = RKNN(verbose=args.verbose)

    try:
        # 2) Pre-process config
        print('--> Configuring model')
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=args.platform,
        )

        # 3) Load ONNX model
        print('--> Loading model')
        if rknn.load_onnx(model=args.model) != 0:
            raise RuntimeError('Failed to load ONNX model')

        # 4) Build model
        print(f'--> Building model (quantization={do_quant})')
        if rknn.build(do_quantization=do_quant, dataset=args.dataset) != 0:
            raise RuntimeError('Failed to build model')

        # 5) Export RKNN model
        print('--> Exporting RKNN model')
        if rknn.export_rknn(output_path) != 0:
            raise RuntimeError('Failed to export RKNN model')

        print(f'Model exported to: {output_path}')

    finally:
        # 6) Release
        rknn.release()
