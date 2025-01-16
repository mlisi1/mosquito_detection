#!/bin/bash
sudo docker run --gpus all -v ~/CIB/mosquito_detection/trainings/:/workspace/trainings -v /tmp/.X11-unix/:/tmp/.X11-unix:rw -it mosquito_detection
