import argparse
import os.path as osp

from seismogen.config import system_config


def get_parser():
    parser = argparse.ArgumentParser("Main argument parser")

    # Main experiment params
    parser.add_argument("--task_name", default="seg")
    parser.add_argument("--random_state", type=int, default=24)

    # Common model & dataloader params
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")

    # Dataloader params
    parser.add_argument("--data_dir", default=osp.join(system_config.data_dir, "raw"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--letterbox", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--augmentation_intensity", default=None)
    parser.add_argument("--gauss_limit", type=float, default=0.0)
    parser.add_argument("--val_size", type=float, default=0.2)

    # Seg dataloader params
    parser.add_argument("--track_num", type=int, default=1)
    parser.add_argument("--sample_type", default="random")
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--balancing_coeff", default='{"track1": 1.0, "track2": 1.0}')

    # Seg model params
    parser.add_argument("--seg_model_arch", default="Unet")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained_weights", default="imagenet")
    parser.add_argument("--num_classes", type=int, default=7)

    # Prediction params
    parser.add_argument(
        "--tta_type", default=None, help="type of tta importable from pytorch_toolbelt.inference.tta module"
    )
    parser.add_argument("--fill_holes", action="store_true")
    parser.add_argument("--biggest_only", action="store_true")

    # Optimizer params
    parser.add_argument("--optimizer_type", default="adamw")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Scheduler params
    parser.add_argument("--lr_scheduler", default=None)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--gamma_factor", type=float, default=0.5)

    # Train params
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--loss_weights", type=float, nargs="+", default=None)
    parser.add_argument("--use_focal", action="store_true")

    return parser
