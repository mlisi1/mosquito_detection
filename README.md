# mosquito_detection

## Building
To use this framework, the [MMDetection](https://github.com/open-mmlab/mmdetection) docker image is needed:
```bash
git clone git@github.com:open-mmlab/mmdetection.git
cd mmdetection/
sudo docker image build -t mmdetection docker/
```
Then, ```nvidia-container-toolkit``` and ```nvidia-docker2``` are needed on the host machine.
```bash
sudo apt install -y nvidia-container-toolkit nvidia-docker2
```
To build the docker file use 
```bash
sudo docker build -t mosquito_detection .
```
The building process includes the download of [MMYolo](https://github.com/open-mmlab/mmyolo) library, necessary to train YOLO networks with a version higher than 3.

---
## Usage
To run the docker and explore its content and scripts you an use:
```bash
sudo docker run --gpus all -v /tmp/.X11-unix/:/tmp/.X11-unix:rw --ipc=host -it mosquito_detection
```
or as an alternative, ```bash run.sh```.
To start a training sessione, an additional argument has to be passed specifying the volume to share with the docker:
```bash
sudo docker run --gpus all -v <absolute/path/to/trainings/dir>:/workspace/training -v /tmp/.X11-unix/:/tmp/.X11-unix:rw mosquito_detection train ARGS
```
`train` is just a macro to execute `mmdetection` train.py script, so the arguments behave the same as mmpretrain scripts.

To use any scripts that launch GUI windows, it's necessary to run ```xhost +``` on the host machine.

Other implemented macros are for `confusion_matrix`, `browse_dataset`, `test`, and `analyze_logs`.
---
## Changes to the base image
Beside the config files for the networks and datasets, a fer scripts has been added:
+ ```tools/dataset_converters/yolo2coco.py```: Converts datasets from YOLO annotations format to COCO annotations
+ ```tools/dataset_converters/combine_coco_datasets.py```: Combines two COCO formatted datasets into a single one

A small change has been made to ```mmdet/__init__.py``` which caused a dependancy conflict which prevented the correct use of the framework.

Some of the default tools have been found not to work correctly, or not correctly implementing the input arguments.
+ `confusion_matrix.py` has been modified to accept `--plt-cfg`, which specifies the matplotlib plot configuration
+ `analyze_logs.py` has been modified to correctly process the `--eval-interval` and `--start-epoch` arguments
+ `dataset_class_histogram.py` has been added. It analyzes COCO formatted datasets and creates bar plots to visualize the number of instances per classes.

The post_training folder has been added to the tools available. This folder contains some bash scripts to automate post trining network analisys, providing a way to execute some scripts on every folder present in `/workspace/trainings`.
At the moment, it automates the test.py, analyze_logs.py and confusion_matrix.py scripts.
