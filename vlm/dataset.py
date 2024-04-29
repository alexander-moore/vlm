"""
Dataset and dataloading for vision-language models.

We need to do a few things here:
load image-text pairs
format them

Do we do this here or in the model?
the model needs a way to take a set [dict1, dict2, ...] where dict_i = {text, imgae}
Then format these into token sequences, pad them, and collate them into batches 

So I guess the dataset can't really do that. Needs to be in the model to access the tokenizers, padding embedding etc?
"""

"""
Sample Data
-----------

Sample data is borrowed from the `cppe-5` dataset. I use this data since it has images, string labels, and some interesting annotations such as bboxes we may enjoy using.
It is also a small, reasonable size for testing

Quote: https://huggingface.co/docs/datasets/en/object_detection
The dataset has the following fields:

image: PIL.Image.Image object containing the image.
image_id: The image ID.
height: The image height.
width: The image width.
objects: A dictionary containing bounding box metadata for the objects in the image:
id: The annotation id.
area: The area of the bounding box.
bbox: The object’s bounding box (in the coco format).
category: The object’s category, with possible values including Coverall (0), Face_Shield (1), Gloves (2), Goggles (3) and Mask (4).
"""

from datasets import load_dataset
import matplotlib.pyplot as plt
import warnings
import random
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Resize
from random import choice

def get_pokemon_dataset():
    """
    Get image-caption pokemon dataset. use this as val
    """
    return load_dataset("lambdalabs/pokemon-blip-captions")

def get_coco_dataset(mode = 'train'):
    """
    Abcd
    """
    coco_dataset = dset.CocoDetection(root = f'/data/coco2017/{mode}2017',
                                        annFile = f'/data/coco2017/annotations/captions_{mode}2017.json'
                                        )
    return Coco_Wrapper(coco_dataset)
    
class Coco_Wrapper(Dataset):
    def __init__(self, coco_dataset):
        self.dataset = coco_dataset
        #self.transforms = Compose(ToTensor(), Resize((256,256), antialias=True))
        self.totensor = ToTensor()
        self.resize = Resize((256, 256), antialias=True)
        self.len = len(coco_dataset)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        """
        Get and transform items
        """
        img, target = self.dataset[idx]
        image = self.totensor(img)
        #image = self.resize(image)
        
        caption = choice(target)['caption']
        
        sample = {'image': image,
                  'caption': caption}
        return sample
        
        

from torch.utils.data import DataLoader
if __name__ == '__main__':
    """
    This function is used to test datasets returns the correct
    """
    # Load a test dataset
    dataset = get_coco_dataset()
    
    batch = dataset[0]
    print(batch['image'].shape, batch['caption'])
    
    dataloader = DataLoader(dataset, batch_size = 2)
    batch = next(iter(dataloader))
    print(batch['image'].shape, batch['caption'])
    
