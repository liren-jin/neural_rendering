## Environment Setup
```commandline
git clone git@github.com:liren-jin/neural_rendering.git
cd neural_rendering
conda env create -f environment.yaml
conda activate neural_rendering
```

## Dataset
- download [DTU dataset](https://phenoroam.phenorob.de/file-uploader/download/public/953455041-dtu_dataset.zip) and [Shapenet dataset](https://phenoroam.phenorob.de/file-uploader/download/public/731944960-shapenet.zip) to data/dataset folder.

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
