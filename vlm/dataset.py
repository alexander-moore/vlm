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

# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
    
#     ds = load_dataset("lambdalabs/pokemon-blip-captions")
#     ds

# idx = random.randint(0, 40)

# example = ds['train'][idx]

# plt.imshow(example['image'])
# plt.show()

#print(exampl#e)

def get_coco_dataset():
    """
    get image captions dataset - use this as train
    """
    import fiftyone
    dataset = fiftyone.zoo.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["captioons"],
        max_samples=50,
    )
    

    # Visualize the dataset in the FiftyOne App
    #session = fiftyone.launch_app(dataset)
    
    return dataset

def get_pokemon_dataset():
    """
    Get image-caption pokemon dataset. use this as val
    """
    return load_dataset("lambdalabs/pokemon-blip-captions")

import torch
import fiftyone.utils.coco as fouc
from PIL import Image
class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.
    
    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the 
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """

    def __init__(
        self,
        fiftyone_dataset,
    ):
        self.samples = fiftyone_dataset
        self.img_paths = self.samples.values("filepath")

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")
        
        print(img_path, sample, metadata)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        print(img.shape)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

from torch.utils.data import DataLoader
if __name__ == '__main__':
    # Load a test dataset
    dataset = get_coco_dataset()
    dataset = FiftyOneTorchDataset(dataset)
    dataloader = DataLoader(dataset, batch_size = 1)
    
    print(dataset, dir(dataset))
    
    item = next(iter(dataloader))
    print(item)