#!/bin/bash

# Root directory path
root_dir="/home/cc/mlProfiler/tests/mps/ccv100_logs/baseline/inf"

# Loop through each subdirectory recursively starting from the root
find "$root_dir" -type d -name "FREQ*" | while read -r freq_dir; do
    # Get the frequency part (FREQXXX)
    freq=$(basename "$freq_dir")
    
    # Get the parent directory path of the FREQ directory
    parent_dir=$(dirname "$freq_dir")
    
    # Determine the new directory structure
    new_dir="$root_dir/$freq/$(echo $parent_dir | sed "s|$root_dir/||")"
    
    # Create the new directory structure
    sudo mkdir -p "$new_dir"
    
    # Move the contents from the old location to the new one
    sudo mv "$freq_dir"/* "$new_dir"
    
    # Remove the empty FREQ directory
    sudo rmdir "$freq_dir"
done
