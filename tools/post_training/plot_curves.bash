#!/bin/bash

# Loop through each directory in the current folder
for dir in /workspace/trainings/*/ ; do
    # Check if it's a directory
    # dir=${dir%/}

   
    #Check if it's a directory
    if [ -d "$dir" ]; then
        # echo "Processing folder: $dir"

        subdirs=($(find "$dir" -maxdepth 1 -type d -name "20*"))

        if [ ${#subdirs[@]} -eq 0 ]; then
            echo "No subdirectories starting with '20' found in $dir. Skipping."
            continue
        fi

        target_folder=""
        target_json=""
        
        for subdir in "${subdirs[@]}"; do
            vis_data_dir="$subdir/vis_data"

            
            if [ -d "$vis_data_dir" ]; then
                json_file="$vis_data_dir/$(basename "$subdir").json"
                scalars_file="$vis_data_dir/scalars.json"

                if [[ -f "$json_file" && -f "$scalars_file" ]]; then
                    target_folder="$subdir"
                    target_json="$json_file"
                    break  # Stop searching once the correct folder is found
                fi
            fi
        done

        if [ -n "$target_folder" ]; then
            echo "Saving plots for loss and mAP for network $(basename "$dir")"
            analyze_logs plot_curve $target_json --eval-interval 200 --key loss loss_cls loss_bbox --legend loss loss_cls loss_bbox --out $dir/loss.png
            analyze_logs plot_curve $target_json --eval-interval 200 --key bbox_mAP bbox_mAP_50 --legend bbox_mAP bbox_mAP_50 --out $dir/mAP.png
        fi


    fi
done
