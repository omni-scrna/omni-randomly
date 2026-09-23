"""Main functions for the OmniBenchmark module."""

from pathlib import Path


def process_data(args):
    """Process data using parsed command-line arguments.

    Args:
        args: Parsed arguments from argparse containing:
            - output_dir: Output directory path
            - name: Module name
            - rawdata_h5ad: Input files for rawdata_h5ad (CLI: --rawdata_h5ad)
            - normalized_h5: Input files for normalized_h5 (CLI: --normalized_h5)
            - filtered_cellids: Input files for filtered_cellids (CLI: --filtered_cellids)
            - filtered_featureids: Input files for filtered_featureids (CLI: --filtered_featureids)
            - properties_info: Input files for properties_info (CLI: --properties_info)

    Note: Input IDs with dots (e.g., 'data.raw') are converted to underscores
          in Python variable names (e.g., 'data_raw') but preserve dots in CLI args.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing module: {args.name}")

    # Access stage inputs
    rawdata_h5ad_files = args.rawdata_h5ad
    print(f"  rawdata_h5ad: {rawdata_h5ad_files}")
    normalized_h5_files = args.normalized_h5
    print(f"  normalized_h5: {normalized_h5_files}")
    filtered_cellids_files = args.filtered_cellids
    print(f"  filtered_cellids: {filtered_cellids_files}")
    filtered_featureids_files = args.filtered_featureids
    print(f"  filtered_featureids: {filtered_featureids_files}")
    properties_info_files = args.properties_info
    print(f"  properties_info: {properties_info_files}")

    # TODO: Implement your processing logic here
    # Example: Read inputs, process, write outputs

    # Write a simple output file
    output_file = output_dir / f"{args.name}_result.txt"
    with open(output_file, 'w') as f:
        f.write(f"Processed module: {args.name}\n")
        f.write(f"rawdata_h5ad: {len(rawdata_h5ad_files)} file(s)\n")
        f.write(f"normalized_h5: {len(normalized_h5_files)} file(s)\n")
        f.write(f"filtered_cellids: {len(filtered_cellids_files)} file(s)\n")
        f.write(f"filtered_featureids: {len(filtered_featureids_files)} file(s)\n")
        f.write(f"properties_info: {len(properties_info_files)} file(s)\n")

    print(f"Results written to: {output_file}")
