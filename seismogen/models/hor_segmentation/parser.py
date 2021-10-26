import argparse
import os.path as osp

from seismogen.config import system_config


def get_parser():
    parser = argparse.ArgumentParser("Main argument parser")

    # Main experiment params
    parser.add_argument("--task_name", default="seg")
    parser.add_argument("--random_state", type=int, default=24)

    # Common model & dataloader params
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")

    # Selection datasets params
    parser.add_argument("--train_datasets", type=str, nargs="+", default=["f3_demo"])
    parser.add_argument("--test_datasets", type=str, nargs="*", default=["penobscot"])
    parser.add_argument("--view_datasets", type=str, nargs="*", default=[])

    # Dataset params
    parser.add_argument("--data_dir", default=system_config.data_dir)
    parser.add_argument("--json_path", default=osp.join("processed", "volume.json"))
    parser.add_argument("--target_type", type=str, default="horizons")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--letterbox", action="store_true")
    parser.add_argument("--augmentation_intensity", default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--split_type", type=str, default="first")

    # Dataloader params
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sample_type", default="random")

    # Seg model params
    parser.add_argument("--enable_gan", action="store_true")
    parser.add_argument("--seg_model_arch", default="Unet")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained_weights", default="imagenet")
    parser.add_argument("--num_classes", type=int, default=1)

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
    parser.add_argument("--evaluate_before_training", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)

    # GAN - Generator parameters
    parser.add_argument("--style_dim", type=int, default=100)

    # GAN - Train params
    parser.add_argument("--disc_clip_value", type=float, default=1e-2)
    parser.add_argument("--n_critic", type=int, default=1)

    return parser
