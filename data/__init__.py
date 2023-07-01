from .dtu import DTUDataModule
from .shapenet import ShapenetDataModule


def get_data(cfg):
    dataset_name = cfg["name"]

    if dataset_name == "dtu":
        print(f"loading dtu dataset \n")
        return DTUDataModule(cfg)
    elif dataset_name == "shapenet":
        print(f"loading shapenet dataset \n")
        return ShapenetDataModule(cfg)
    else:
        RuntimeError("dataset is not implemeneted!")
