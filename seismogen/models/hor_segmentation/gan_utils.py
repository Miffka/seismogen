import torch
from torch import autograd


def compute_gradient_penalty(
    netD: torch.nn.Module, real_data: torch.Tensor, fake_data: torch.Tensor, real_fake_class_idx: int = 1
) -> torch.Tensor:
    batch_size = real_data.size(0)
    # Sample Epsilon from uniform distribution
    eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    eps = eps.expand_as(real_data)

    # Interpolation between real data and fake data.
    interpolation = eps * real_data + (1 - eps) * fake_data

    fake_imgs_classify = getattr(netD, "classification_head", None) is not None

    # get logits for interpolated images
    if fake_imgs_classify:
        interp_logits = netD(interpolation)[1]
    else:
        interp_logits = netD(interpolation)[:, real_fake_class_idx].unsqueeze(1)
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute and return Gradient Norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)
