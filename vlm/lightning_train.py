"""
Use lightning for DDP training of VLM

"""

import lightning as L
from lit_vlm import VLM_LitModel
from vlm import build_vlm
import argparse

parser = argparse.ArgumentParser(description='VLM Training Settings')

parser.add_argument('--gpu_ids', help='Comma-separated list of GPU Numbers to use', 
                    default='0', type=str)

parser.add_argument('--version_name', help = 'version name in save_dir', 
                    type = str, default = 'test')

args = parser.parse_args()


model = build_vlm()
lit_model = VLM_LitModel(model = model)

from lightning.pytorch.strategies import DDPStrategy


# Added gradient clipping
from dataset import get_coco_dataset
from torch.utils.data import DataLoader

gpu_ids = [int(i) for i in args.gpu_ids.split(',')]

train_dataset = get_coco_dataset(mode='train')
train_dataloader = DataLoader(train_dataset, batch_size = 1)

print('im validating on testing dataset!!')
val_dataset = get_coco_dataset(mode = 'train') # mode = val BTW!
val_dataloader = DataLoader(val_dataset, batch_size = 1)

from lightning.pytorch.loggers import TensorBoardLogger
save_dir = 'logs'
version_name = args.version_name

logger = TensorBoardLogger(save_dir=save_dir, version=version_name, name="trackers")
    
trainer = L.Trainer(accelerator="gpu", devices=gpu_ids, 
                    gradient_clip_val = 1,
                     max_steps = 60000)


trainer.fit(model=lit_model, train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader)
