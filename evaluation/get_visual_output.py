import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.pretrained_model import PretrainedModel
from data import get_data
from utils import parser, util
import yaml
from dotmap import DotMap
import torch
import warnings
import numpy as np
import imageio
from datetime import datetime

warnings.filterwarnings("ignore")


def main():
    """
    given scene index, reference index and novel view index,
    this script outputs ground truth, reference, uncertainty, depth and RGB images.
    used for sanity check.
    """

    args = parser.parse_args(visual_args)
    log_path = os.path.join("logs", args.model_name)

    assert os.path.exists(log_path), "experiment does not exist"
    with open(f"{log_path}/training_setup.yaml", "r") as config_file:
        cfg = yaml.safe_load(config_file)

    checkpoint_path = os.path.join(log_path, "checkpoints", "best.ckpt")
    assert os.path.exists(checkpoint_path), "checkpoint does not exist"
    ckpt_file = torch.load(checkpoint_path)

    gpu_id = list(map(int, args.gpu_id.split()))
    device = util.get_cuda(gpu_id[0])

    model = PretrainedModel(cfg["model"], ckpt_file, device, gpu_id)

    datamodule = get_data(cfg["data"])
    dataset = datamodule.load_dataset("val")
    z_near = dataset.z_near
    z_far = dataset.z_far

    scene_idx = args.scene_idx
    ref_idx = list(map(int, args.ref_idx.split()))
    target_idx = args.target_idx

    data_instance = dataset.__getitem__(scene_idx)
    scene_title = data_instance["scan_name"]
    print(f"visual test on {scene_title}")

    images = data_instance["images"].to(device)
    images_0to1 = images * 0.5 + 0.5
    _, _, H, W = images.shape
    focal = data_instance["focal"].to(device)
    c = data_instance["c"].to(device)
    poses = data_instance["poses"].to(device)

    with torch.no_grad():
        model.network.encode(
            images[ref_idx].unsqueeze(0),
            poses[ref_idx].unsqueeze(0),
            focal.unsqueeze(0),
            c.unsqueeze(0),
        )

        novel_pose = poses[target_idx]
        target_rays = util.gen_rays(
            novel_pose.unsqueeze(0), W, H, focal, z_near, z_far, c
        )
        target_rays = target_rays.reshape(1, H * W, -1)
        predict = DotMap(model.renderer_par(target_rays))
        rgb = predict.rgb[0].cpu().numpy().reshape(H, W, 3)
        uncertainty = predict.uncertainty[0].cpu().numpy().reshape(H, W)
        depth = predict.depth[0].cpu().numpy().reshape(H, W)

        gt = images_0to1[target_idx].permute(1, 2, 0).cpu().numpy()
        ref_images = images_0to1[ref_idx].permute(0, 2, 3, 1).cpu().numpy()

        ref_images = np.hstack((*ref_images,))
        error = np.abs(rgb - gt)

    error_map = util.error_cmap(error)
    uncertainty_map = util.unc_cmap(uncertainty)
    depth_map = util.depth_cmap(depth, z_near, z_far)

    experiment_path = os.path.join(
        "experiments",
        args.model_name,
        "visual_experiment",
        datetime.now().strftime("%d-%m-%Y-%H-%M"),
    )

    os.makedirs(experiment_path)

    imageio.imwrite(
        f"{experiment_path}/{scene_title}_reference_images_{ref_idx}.jpg",
        (ref_images * 255).astype(np.uint8),
    )
    imageio.imwrite(
        f"{experiment_path}/{scene_title}_rgb_{target_idx}.jpg",
        (rgb * 255).astype(np.uint8),
    )
    imageio.imwrite(
        f"{experiment_path}/{scene_title}_depth_{target_idx}.jpg", depth_map
    )
    imageio.imwrite(
        f"{experiment_path}/{scene_title}_uncertainty_{target_idx}.jpg", uncertainty_map
    )
    imageio.imwrite(
        f"{experiment_path}/{scene_title}_error_{target_idx}.jpg", error_map
    )
    imageio.imwrite(
        f"{experiment_path}/{scene_title}_ground_truth.jpg", (gt * 255).astype(np.uint8)
    )


def visual_args(parser):
    """
    Parse arguments for novel view synthesis setup.
    """

    # mandatory arguments
    parser.add_argument(
        "--model_name",
        "-M",
        type=str,
        required=True,
        help="model name of pretrained model",
    )

    parser.add_argument(
        "--scene_idx",
        "-si",
        type=int,
        required=True,
        help="scene index in DTU validation split",
    )

    parser.add_argument(
        "--ref_idx",
        "-ri",
        type=str,
        required=True,
        help="reference view index, space delimited",
    )

    parser.add_argument(
        "--target_idx", "-ti", type=int, required=True, help="target view index"
    )

    # arguments with default values
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU(s) to use, space delimited"
    )
    return parser


if __name__ == "__main__":
    main()
