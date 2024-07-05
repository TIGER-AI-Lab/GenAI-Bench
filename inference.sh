#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <GPU_ID> <MODEL_NAME> [<TASK_ID>]"
    exit 1
fi

# Assign the input arguments to variables
GPU_ID=$1
MODEL_NAME=$2
TASK_ID=$3

# Export the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Define a function to run inference for a given task
run_inference() {
    local task=$1
    nohup python inference.py --task "$task" --model_name "$MODEL_NAME" > "${MODEL_NAME}_${task}.out" 2>&1 &
}

# If TASK_ID is specified, map it to the corresponding task and run inference
if [ -n "$TASK_ID" ]; then
    case $TASK_ID in
        0)
            TASK="image_generation"
            ;;
        1)
            TASK="image_edition"
            ;;
        2)
            TASK="video_generation"
            ;;
        *)
            echo "Invalid TASK_ID. Use 0 for image_generation, 1 for image_editing, or 2 for video_generation."
            exit 1
            ;;
    esac
    run_inference $TASK
else
    # If TASK_ID is not specified, run inference for all tasks
    run_inference "image_generation"
    run_inference "image_edition"
    run_inference "video_generation"
fi
