#%% 
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split

labeled_dir = "D:/Images_segmentation/Ellipse/pseudo_training/6_images_pt_512"
mask_dir = "D:/Images_segmentation/Ellipse/pseudo_training/6_masks_pt_512"
unlabeled_dir = "D:/Images_nanomax/Images/unlabeled_images_512_t1_pt"

class Labeled_Dataset(Dataset):
    def __init__(self, image_list, mask_list):
        self.image_list = image_list
        self.mask_list = mask_list

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        mask = self.mask_list[index]
        image = torch.load(image)
        mask = torch.load(mask)
        return image, mask
        
class Unlabeled_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        image = torch.load(image)
        return image


def get_dataloaders(batch_size, unlabeled=False, labeled_dir=labeled_dir, mask_dir=mask_dir, unlabeled_dir=unlabeled_dir):
    labeled_list = glob.glob(labeled_dir + '/*.pt')
    mask_list = glob.glob(mask_dir + '/*.pt' )
    train_images, eval_images, train_masks, eval_masks = train_test_split(labeled_list, mask_list, test_size=0.2)
    train_dataset = Labeled_Dataset(train_images, train_masks)
    eval_dataset = Labeled_Dataset(eval_images, eval_masks)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    if unlabeled:
        unlabeled_list = glob.glob(unlabeled_dir + '/*.pt')
        unlabeled_dataset = Unlabeled_Dataset(unlabeled_list)
        unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
        return train_dataloader, eval_dataloader, unlabeled_dataloader
    else:
        return train_dataloader, eval_dataloader
    

