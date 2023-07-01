## Environment Setup
```commandline
git clone git@github.com:liren-jin/neural_rendering.git
cd neural_rendering
conda env create -f environment.yaml
conda activate neural_rendering
```

## DTU Dataset
download DTU dataset from https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR

copy lst files in lst folder to <data root>/rs_dtz_4/DTU folder

## Train on DTU Dataset
for starting training process:
```commandline
python train.py -M <model name>
```
continue traing:
```commandline
python train.py -M <model name> --resume
```
visualize training progress via:
``` commandline
tensorboard --logdir <project dir>/logs/<model name>
```
