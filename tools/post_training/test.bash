#!/bin/bash

# Loop through each directory in the current folder
for dir in /workspace/trainings/*/ ; do
    # Check if it's a directory
    # dir=${dir%/}

   
    #Check if it's a directory
    if [ -d "$dir" ]; then
        echo "Processing folder: $dir"

        # Find the first (only) Python file
        py_file=$(ls "$dir"/*.py 2>/dev/null | head -n 1)

        

        # Find the first file that starts with "best"
        best_file=$(ls "$dir"/best* 2>/dev/null | head -n 1)

        #Check if both files exist before running the command
        if [[ -f "$py_file" && -f "$best_file" ]]; then
            echo "Running: test_nn $py_file $best_file --out $dir/results.pkl"
            test_nn "$py_file" "$best_file" --out "$dir/results.pkl" --show-dir "$dir/test_imgs/" --cfg-options visualizer.type=mmdet.DetLocalVisualizer >> $dir/results.txt
        else
            echo "Skipping $dir: Required files not found."
        fi
    fi
done
