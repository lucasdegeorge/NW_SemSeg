#%% 
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from monai.losses import DiceLoss
import math

with open("parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]
    device = torch.device(device)

def supervised_loss(pred, target, mode):
    assert pred.requires_grad == True and target.requires_grad == False, "Error in requires_grad - supervised"
    assert pred.size() == target.size(), "pred and target must have the same size, must be (batch_size * num_classes * H * W)"

    if mode == "CE":
        pred = F.softmax(pred, dim=1)
        return F.cross_entropy(pred, target, reduction='mean')
    elif mode == "DICE":
        dice_loss = DiceLoss(reduction='sum', softmax=True)
        return dice_loss(pred, target)
    elif mode == "DICE-CE":
        dice = DiceLoss(reduction='sum', softmax=True)
        return dice(pred, target) + F.cross_entropy(F.softmax(pred, dim=1), target, reduction='mean')
    else:
        ValueError("Invalid value for mode. Must be in ['CE', 'DICE', DICE-CE]")


def eval_loss(pred, target, mode):
    assert pred.requires_grad == False and target.requires_grad == False, "Error in requires_grad - eval"
    assert pred.size() == target.size(), "pred and target must have the same size, must be (batch_size * num_classes * H * W)"

    if mode == "CE":
        pred = F.softmax(pred, dim=1)
        return F.cross_entropy(pred, target)
    elif mode == "DICE":
        dice_loss = DiceLoss(reduction='sum', softmax=True)
        return dice_loss(pred, target)
    elif mode == "DICE-CE":
        dice = DiceLoss(reduction='sum', softmax=True)
        return dice(pred, target) + F.cross_entropy(F.softmax(pred, dim=1), target, reduction='mean')
    else:
        ValueError("Invalid value for mode. Must be in ['CE', 'DICE', DICE-CE]")


def unsupervised_loss(pred, target, mode):
    assert pred.requires_grad == True and target.requires_grad == False, "Error in requires_grad"
    assert pred.size() == target.size(), "pred and target must have the same size, must be (batch_size * num_classes * H * W)"

    if mode == "dice":
        dice_loss = DiceLoss(reduction='sum', softmax=True)
        return dice_loss(pred, target)
    if mode == "mse":
        pred = F.softmax(pred, dim=1)
        return F.mse_loss(pred, target, reduction='mean')
    if mode == "kl":
        pred = F.log_softmax(pred, dim=1)
        return F.kl_div(pred, target, reduction='mean')
    if mode == "js":
        M = (F.softmax(pred, dim=1) + target) * 0.5
        kl_P = F.kl_div(F.log_softmax(pred, dim=1), M, reduction='mean')
        kl_Q = F.kl_div(torch.log(target + 1e-5), M, reduction='mean')
        return (kl_P + kl_Q) * 0.5
    else:
        raise ValueError("Invalid value for mode. Must be in ['dice', 'mse', 'kl', 'js']")


# Weight ramp up for unsupervised loss 
def weight_ramp_up(current_step, rampup_length, max_value):
    """Generates the value of 'w' based on a sigmoid ramp-up curve."""
    if rampup_length == 0:
        return max_value
    else:
        current_step = max(0.0, min(current_step, rampup_length))
        phase = 1.0 - current_step / rampup_length
        value = max_value * (math.exp(-5.0 * phase * phase))
        return value
