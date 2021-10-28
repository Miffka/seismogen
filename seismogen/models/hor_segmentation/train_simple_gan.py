import os
import os.path as osp
import time
from typing import Dict, Optional

import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from seismogen.config import system_config
from seismogen.data.segy.dataloaders import init_dataloaders
from seismogen.models.fix_seeds import fix_seeds
from seismogen.models.hor_segmentation.gan_utils import compute_gradient_penalty
from seismogen.models.hor_segmentation.network.network import load_net
from seismogen.models.hor_segmentation.parser import get_parser
from seismogen.models.hor_segmentation.utils import visualize_masks  # noqa F401
from seismogen.models.hor_segmentation.utils import define_losses, eval_model
from seismogen.models.train_utils import define_optimizer, define_scheduler
from seismogen.torch_config import torch_config


def wgan_loss_d(real_disc: torch.Tensor, fake_disc: torch.Tensor) -> torch.Tensor:
    return -torch.mean(real_disc) + torch.mean(fake_disc)


def wgan_loss_g(fake_disc: torch.Tensor) -> torch.Tensor:
    return -torch.mean(fake_disc)


def train_one_epoch(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    fixed_noise: torch.Tensor,
    scheduler_g: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_d: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    predicted_class_idx: int = 0,
    real_fake_class_idx: int = 1,
    epoch_num: int = 0,
    clip_value: float = 0.01,
    n_critic: int = 1,
    fp16_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    writer: Optional[SummaryWriter] = None,
) -> float:

    loss_unsup_disc = wgan_loss_d
    loss_unsup_gen = wgan_loss_g
    loss_sup_ce, loss_sup_d = define_losses(reduction="mean")

    total_samples = len(dataloaders["train"])
    progress_bar = tqdm.tqdm(enumerate(dataloaders["train"]), desc=f"Train epoch {epoch_num}", total=total_samples)
    batch_size = dataloaders["train"].batch_size
    style_dim = generator.style_dim

    generator.train()
    discriminator.train()

    fake_imgs_classify = getattr(discriminator, "classification_head", None) is not None

    for batch_idx, sample in progress_bar:
        optimizer_d.zero_grad()

        calc_sup_loss = True

        with torch.cuda.amp.autocast(enabled=fp16_scaler is not None):
            z = torch.randn((batch_size, style_dim)).to(torch_config.device)

            generated = generator(z)
            predict_d_fake = discriminator(generated.detach())
            predict_d_real = discriminator(sample["image"].to(torch_config.device))

            target_types = np.asarray(sample["target_type"])
            valid_ids = target_types != 0

            if valid_ids.sum() == 0:
                calc_sup_loss = False

            if calc_sup_loss:
                if fake_imgs_classify:
                    loss_sup_ce_v = loss_sup_ce(
                        predict_d_real[0][valid_ids],
                        sample["target"][valid_ids].to(torch_config.device),
                    )
                    loss_sup_d_v = loss_sup_d(
                        predict_d_real[0][valid_ids],
                        sample["target"][valid_ids].to(torch_config.device),
                    )

                else:
                    loss_sup_ce_v = loss_sup_ce(
                        predict_d_real[valid_ids, predicted_class_idx].unsqueeze(1),
                        sample["target"][valid_ids].to(torch_config.device),
                    )
                    loss_sup_d_v = loss_sup_d(
                        predict_d_real[valid_ids, predicted_class_idx].unsqueeze(1),
                        sample["target"][valid_ids].to(torch_config.device),
                    )

            else:
                loss_sup_ce_v = torch.tensor(0, device=torch_config.device)
                loss_sup_d_v = torch.tensor(0, device=torch_config.device)

            if fake_imgs_classify:
                loss_unsup_disc_v = loss_unsup_disc(predict_d_real[1], predict_d_fake[1])

            else:
                loss_unsup_disc_v = loss_unsup_disc(
                    predict_d_real[:, real_fake_class_idx], predict_d_fake[:, real_fake_class_idx]
                )

            if clip_value == 0:
                grad_pen = compute_gradient_penalty(
                    discriminator, real_data=sample["image"].to(torch_config.device), fake_data=generated
                )
            else:
                grad_pen = torch.tensor([0]).to(torch_config.device)

            loss_d = loss_sup_ce_v + loss_sup_d_v + loss_unsup_disc_v + grad_pen

        if fp16_scaler is not None:
            fp16_scaler.scale(loss_d).backward()
            fp16_scaler.step(optimizer_d)
            fp16_scaler.update()
        else:
            loss_d.backward()
            optimizer_d.step()

        # Clip weights of discriminator
        if clip_value != 0:
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # Train generator every n_critic steps
        if batch_idx % n_critic == 0:
            optimizer_g.zero_grad()

            with torch.cuda.amp.autocast(enabled=fp16_scaler is not None):
                predict_d_fake = discriminator(generator(z))
                # Adversarial loss
                if fake_imgs_classify:
                    loss_g = loss_unsup_gen(predict_d_fake[1])
                else:
                    loss_g = loss_unsup_gen(predict_d_fake[:, real_fake_class_idx])

            if fp16_scaler is not None:
                fp16_scaler.scale(loss_g).backward()
                fp16_scaler.step(optimizer_g)
                fp16_scaler.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            progress_bar.set_description(f"Epoch: {epoch_num} Loss_d: {loss_d.item():.3f} Loss_g: {loss_g.item():.3f}")

            if writer is not None:
                global_step = batch_idx + total_samples * epoch_num
                writer.add_scalar("Loss_train/Disc/Total", round(loss_d.item(), 3), global_step=global_step)
                writer.add_scalar(
                    "Loss_train/Disc/Supervised_CE", round(loss_sup_ce_v.item(), 3), global_step=global_step
                )
                writer.add_scalar(
                    "Loss_train/Disc/Supervised_DICE", round(loss_sup_d_v.item(), 3), global_step=global_step
                )
                writer.add_scalar(
                    "Loss_train/Disc/Unsupervised", round(loss_unsup_disc_v.item(), 3), global_step=global_step
                )
                writer.add_scalar("Loss_train/Gen/Unsupervised", round(loss_g.item(), 3), global_step=global_step)

    val_loss_dict = eval_model(
        discriminator,
        dataloaders["val"],
        epoch_num,
        fp16=fp16_scaler is not None,
        postfix="val",
        writer=writer,
    )

    _ = eval_model(
        discriminator,
        dataloaders["test"],
        epoch_num,
        fp16=fp16_scaler is not None,
        postfix="test",
        writer=writer,
    )

    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(fixed_noise.to(torch_config.device)).cpu()
    visualize_masks(discriminator, fake_imgs, epoch_num, header="generated", writer=writer)

    return val_loss_dict["Loss DICE_val"]


def train_model():
    parser = get_parser()
    args = parser.parse_args()

    assert args.enable_gan, f"GAN should be enabled, got args.enable_gan {args.enable_gan}"

    writer = SummaryWriter(
        osp.join(system_config.log_dir + "_" + args.target_type, f"add_gan {args.enable_gan}", args.task_name)
    )

    dataloaders = init_dataloaders(args)

    fix_seeds(args.random_state)
    (generator, discriminator), state = load_net(args)

    generator.to(torch_config.device)
    discriminator.to(torch_config.device)

    optimizer_g = define_optimizer(args, generator, state, postfix="_gen")
    optimizer_d = define_optimizer(args, discriminator, state, lr_multiplier=args.lr_discriminator_multiplier)
    scheduler_g = define_scheduler(args, optimizer_g, state, postfix="_gen")
    scheduler_d = define_scheduler(args, optimizer_d, state)

    init_epoch = state.get("epoch", 0)
    best_metric = state.get("best_metric", 100)

    save_dir = osp.join(system_config.model_dir, args.task_name)
    os.makedirs(save_dir, exist_ok=True)

    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    else:
        fp16_scaler = None

    if args.evaluate_before_training:
        _ = eval_model(
            discriminator,
            dataloaders["val"],
            init_epoch - 1,
            fp16=fp16_scaler is not None,
            postfix="val",
            writer=writer,
        )
        _ = eval_model(
            discriminator,
            dataloaders["test"],
            init_epoch - 1,
            fp16=fp16_scaler is not None,
            postfix="test",
            writer=writer,
        )

    fix_seeds(args.random_state)
    fixed_noise = torch.randn((args.batch_size, args.style_dim))

    for epoch_num in tqdm.tqdm(range(init_epoch, init_epoch + args.epochs), desc="Epochs"):
        current_metric = train_one_epoch(
            generator,
            discriminator,
            dataloaders,
            optimizer_g,
            optimizer_d,
            fixed_noise,
            scheduler_g,
            scheduler_d,
            epoch_num=epoch_num,
            clip_value=args.disc_clip_value,
            n_critic=args.n_critic,
            fp16_scaler=fp16_scaler,
            writer=writer,
        )
        state = {
            "state_dict": discriminator.state_dict(),
            "state_dict_gen": generator.state_dict(),
            "epoch": epoch_num,
            "best_metric": current_metric,
            "optimizer": optimizer_d.__class__.__name__,
            "optimizer_state": optimizer_d.state_dict(),
            "optimizer_state_gen": optimizer_g.state_dict(),
        }
        torch.save(state, osp.join(save_dir, "last.pth"))

        if current_metric < best_metric:
            best_metric = current_metric
            torch.save(state, osp.join(save_dir, "best.pth"))

        time.sleep(5)


if __name__ == "__main__":
    train_model()
