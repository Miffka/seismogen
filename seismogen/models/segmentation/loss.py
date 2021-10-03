import torch


class DiceLoss(torch.nn.Module):
    def __init__(self, device):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, masks):
        smooth = 1e-15
        prediction = outputs.softmax(dim=1)

        dice_loss = None

        for val in range(0, 7):
            ch_pred = prediction[:, val]
            # ch_pred = ch_pred / (ch_pred + 1e-10)

            intersection = torch.sum(ch_pred * masks[:, val], dim=(1, 2))
            union = torch.sum(ch_pred, dim=(1, 2)) + torch.sum(masks[:, val], dim=(1, 2))
            dice_part = (1 - (2 * intersection + smooth) / (union + smooth)).abs()
            dice_loss = dice_part if dice_loss is None else dice_loss + dice_part

        return dice_loss / 7
