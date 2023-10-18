#%%
import torch
import json
import io
from torchmetrics import JaccardIndex, Accuracy, Dice
import glob

from unet_models import *
from dataloader import *
from display import *

with open("parameters/Unet_parameters.json", 'r') as f:
    arguments = json.load(f)
    device = arguments["device"]

model_folder = "C:/Users/lucas.degeorge/Documents/trained_models"
image_folder = "C:/Users/lucas.degeorge/Documents/Images/eval_images"
mask_folder = "C:/Users/lucas.degeorge/Documents/Images/eval_masks"


def compute_metrics(model, mode, eval_dataloader=None, display=False):
    if eval_dataloader is None:
        labeled_list = glob.glob(image_folder + '/*.pt')
        mask_list = glob.glob(mask_folder + '/*.pt' )
        eval_dataset = Labeled_Dataset(labeled_list, mask_list)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, pin_memory=True)
    
    model.eval()

    metrics = [0, 0, 0]  # [mIoU, pixel_acc, dice]
    jaccard = JaccardIndex(task="multiclass", num_classes=3).to(device)
    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
    dice = Dice(average="macro", num_classes=3).to(device)

    i = 0 
    for x, mask in eval_dataloader:
        # print(i)
        if mode=="CCT":  pred = model(x.to(device), eval=True)["output_l"]
        elif mode=="UNet":  pred = model(x.to(device), eval=True)

        pred = pred.permute(0,2,3,1)
        pred = torch.softmax(pred, dim=-1)
        pred = torch.argmax(pred, dim=-1)

        if display:
        # if i%19==0:
            display_save_image_with_mask(x[0], pred)
            # display_save_image_mask_overlaid(x[0], pred)

        mask = mask.to(device).permute(0,2,3,1)
        mask = torch.argmax(mask, dim=-1) 

        metrics[0] += jaccard(pred, mask)
        metrics[1] += accuracy(pred, mask)
        metrics[2] += dice(pred, mask)

        i+=1

    metrics = [ x.item() / len(eval_dataloader) for x in metrics ]

    return metrics

