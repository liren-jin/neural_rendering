from .dtu import DTUDataModule
from .shapenet import ShapenetDataModule


def get_data(cfg):
    dataset_name = cfg["name"]
    print(f"loading {dataset_name} dataset \n")

    if dataset_name == "dtu":
        return DTUDataModule(cfg)
    elif dataset_name == "shapenet":
        return ShapenetDataModule(cfg)
    else:
        RuntimeError("dataset is no implemeneted!")
