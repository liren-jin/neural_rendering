import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trainer import Trainer
from model import get_network, get_renderer
from data import get_data
from utils import parser
import click
import yaml
import warnings

warnings.filterwarnings("ignore")


def main():
    args = parser.parse_args(training_args)
    log_path = os.path.join(args.logs_path, args.model_name)

    # start training from scratch
    if not args.resume:
        if os.path.exists(log_path):
            click.confirm(
                "experiment/model already exits, start training from scratch? "
                "(will overwrite log files and checkpoints); "
                "otherwise, use --resume in command line",
                abort=True,
            )

        os.makedirs(log_path, exist_ok=True)
        os.makedirs(os.path.join(log_path, args.checkpoints_path), exist_ok=True)

        # load model configuration
        assert os.path.exists(args.model_cfg_path)
        with open(args.model_cfg_path, "r") as config_file:
            model_cfg = yaml.safe_load(config_file)

        # load dataset configuration
        assert os.path.exists(args.data_cfg_path)
        with open(args.data_cfg_path, "r") as config_file:
            data_cfg = yaml.safe_load(config_file)

        # load trainer configuration
        assert os.path.exists(args.trainer_cfg_path)
        with open(args.trainer_cfg_path, "r") as config_file:
            trainer_cfg = yaml.safe_load(config_file)

        # record complete configuration including command line args of the experiment
        cfg = {**model_cfg, **data_cfg, **trainer_cfg}
        record_cfg = cfg.copy()
        record_cfg["command_line"] = args.__dict__

        with open(
            f"{args.logs_path}/{args.model_name}/training_setup.yaml", "w"
        ) as config_file:
            yaml.safe_dump(record_cfg, config_file, default_flow_style=False)

    # resume training
    else:
        assert os.path.exists(log_path), "experiment does not exist"
        with open(f"{log_path}/training_setup.yaml", "r") as config_file:
            cfg = yaml.safe_load(config_file)

    datamodule = get_data(cfg["data"])
    network = get_network(cfg["network"])
    renderer = get_renderer(cfg["renderer"])

    trainer = Trainer(args, cfg["trainer"], datamodule, network, renderer)
    trainer.start()


def training_args(parser):
    """
    Parse arguments for training setup.
    """

    # mandatory arguments
    parser.add_argument(
        "--model_name",
        "-M",
        type=str,
        required=True,
        help="unique experiment/model name for training instance",
    )
    parser.add_argument(
        "--data_path", "-D", type=str, required=True, help="data root path"
    )

    # arguments with default values
    parser.add_argument(
        "--model_cfg_path",
        "-MC",
        type=str,
        default="config/model.yaml",
        help="model config file path",
    )
    parser.add_argument(
        "--data_cfg_path",
        "-DC",
        type=str,
        default="config/data.yaml",
        help="data config file path",
    )
    parser.add_argument(
        "--trainer_cfg_path",
        "-TC",
        type=str,
        default="config/training.yaml",
        help="trainer config file path",
    )
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        default="checkpoints",
        help="checkpoints output directory name",
    )
    parser.add_argument(
        "--logs_path", type=str, default="logs", help="logs output directory name"
    )
    parser.add_argument(
        "--print_interval",
        "-pi",
        type=int,
        default=5,
        help="training step interval for printing training metrics",
    )
    parser.add_argument(
        "--save_interval",
        "-si",
        type=int,
        default=1,
        help="epoch interval for saving checkpoints",
    )
    parser.add_argument(
        "--vis_interval",
        "-vi",
        type=int,
        default=1,
        help="epoch interval for visualization",
    )
    parser.add_argument(
        "--vis_repeat",
        "-vp",
        type=int,
        default=10,
        help="repeat visualization for one object with different reference views",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU(s) to use, space delimited"
    )
    parser.add_argument("--resume", action="store_true", help="continue training")
    return parser


if __name__ == "__main__":
    main()
