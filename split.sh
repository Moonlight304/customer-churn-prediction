#!/bin/bash

# Default values
parts=10
prefix="part_"

while getopts "i:p:" opt; do
  case $opt in
    i) input="$OPTARG" ;;
    p) parts="$OPTARG" ;;
    *) echo "Usage: $0 [-i input] [-p parts]"; exit 1 ;;
  esac
done

# Check if input file exists
if [[ ! -f "$input" ]]; then
    echo "Error: File $input not found."
    exit 1
fi

# Define the output directory based on the input file location
input_dir=$(dirname "$(readlink -f "$input")")
filename_with_ext=${input##*/}
filename="${filename_with_ext%.*}"
output_dir="$input_dir/parts_$filename"
mkdir -p "$output_dir"

# Get line counts
total_lines=$(wc -l < "$input")
data_lines=$((total_lines - 1))

# Lines per split (ceiling division)
lines_per_file=$(( (data_lines + parts - 1) / parts ))

# Extract header
header=$(head -n 1 "$input")

# Split data directly into the output directory
# -d: use numeric suffixes (00, 01...)
# -a 3: use 3 digits for suffixes (handles up to 1000 parts)
tail -n +2 "$input" | split -l "$lines_per_file" -d -a 3 - "$output_dir/$prefix"

# Add header back to each chunk
# We loop specifically through the files created in the output_dir
for f in "$output_dir/${prefix}"[0-9]*; do
    # Create the final CSV with the header
    (echo "$header"; cat "$f") > "$f.csv"
    # Remove the temporary split chunk
    rm "$f"
done

echo "Done. Created $parts CSV files in: $output_dir"