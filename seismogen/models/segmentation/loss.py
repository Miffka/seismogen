import torch


class DiceLoss(torch.nn.Module):
    def __init__(self, device: torch.device):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, masks):
        smooth = 1e-15
        prediction = outputs.softmax(dim=1)
        num_classes = outputs.shape[1]
        start_class = num_classes - 7

        dice_loss = None

        for val in range(start_class, num_classes):
            ch_pred = prediction[:, val]
            if masks.ndim == 3:
                ch_mask = (masks == val).float()
            else:
                ch_mask = masks[:, val]

            intersection = torch.sum(ch_pred * ch_mask, dim=(1, 2))
            union = torch.sum(ch_pred, dim=(1, 2)) + torch.sum(ch_mask, dim=(1, 2))
            dice_part = (1 - (2 * intersection + smooth) / (union + smooth)).abs()
            dice_loss = dice_part if dice_loss is None else dice_loss + dice_part

        return dice_loss / 7
