## Environment Setup
```commandline
git clone git@github.com:liren-jin/neural_rendering.git
cd neural_rendering
conda env create -f environment.yaml
conda activate neural_rendering
```

## Dataset
- download DTU dataset and Shapenet dataset from https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR to data/dataset folder.

## Training
for starting training process:
```commandline
python train.py -M <model name> --setup_cfg_path <path to training setup file>
```
continue training:
```commandline
python train.py -M <model name> --setup_cfg_path <path to training setup file> --resume
```
visualize training progress via:
``` commandline
tensorboard --logdir <project dir>/logs/<model name>
```
