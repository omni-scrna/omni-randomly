#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from main import process_data

def parse_args():
    parser = argparse.ArgumentParser(description='OmniBenchmark module')

    # Required by OmniBenchmark
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--name', type=str, required=True,
                       help='Module name/identifier')
    # Stage-specific inputs
    parser.add_argument('--rawdata_h5ad', nargs='+', dest='rawdata_h5ad', required=True,
                       help='Input: rawdata_h5ad')
    parser.add_argument('--normalized_h5', nargs='+', dest='normalized_h5', required=True,
                       help='Input: normalized_h5')
    parser.add_argument('--filtered_cellids', nargs='+', dest='filtered_cellids', required=True,
                       help='Input: filtered_cellids')
    parser.add_argument('--filtered_featureids', nargs='+', dest='filtered_featureids', required=True,
                       help='Input: filtered_featureids')
    parser.add_argument('--properties_info', nargs='+', dest='properties_info', required=True,
                       help='Input: properties_info')

    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Output directory: {args.output_dir}")
    print(f"Module name: {args.name}")
    print(f"rawdata_h5ad: {args.rawdata_h5ad}")
    print(f"normalized_h5: {args.normalized_h5}")
    print(f"filtered_cellids: {args.filtered_cellids}")
    print(f"filtered_featureids: {args.filtered_featureids}")
    print(f"properties_info: {args.properties_info}")

    # TODO: Implement your module logic
    # Process the data using main function
    process_data(args)

if __name__ == "__main__":
    main()
