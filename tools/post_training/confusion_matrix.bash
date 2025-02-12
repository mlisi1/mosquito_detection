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
        prediction=$(ls "$dir"/*.pkl 2>/dev/null | head -n 1)


        #Check if both files exist before running the command
        if [[ -f "$py_file" && -f "$prediction" ]]; then
            echo "Running: confusion_matrix $py_file $prediction  $dir --plt-cfg 0.193 0.26 0.92 0.7 0.2 0.2"
            confusion_matrix "$py_file" "$prediction" "$dir" --plt-cfg 0.193 0.26 0.92 0.7 0.2 0.2
        else
            echo "Skipping $dir: Required files not found."
        fi
    fi
done
