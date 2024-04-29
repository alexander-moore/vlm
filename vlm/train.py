"""
Train a vlm
"""

device = 7
# def batch_to_device(batch):
#     for key, value in batch.items():
#         try:
#             batch[key] = value.to(device)
#             print(key, value)
#         except:
#             pass

def train_model(model, n_epochs):
    model = model.to(device)
    model.train()
    optimizer.train()
    
    for _ in range(n_epochs):
        losses = []
        for bi, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            #batch = batch_to_device(batch)
            batch['image'] = batch['image'].to(device)
            
            logits, loss = model.forward(batch)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.data.item())
            if bi % 100 == 0:
                print(sum(losses) / len(losses))
            
        val_metrics = val_step()
        print(val_metrics)
    
def val_step():
    model.val()
    optimizer.val()
    
    val_metrics = 0
    
    model.train()
    optimizer.train()
    
    return val_metrics

if __name__ == '__main__':
    """
    Train a model
    """
    
    # Model
    from vlm import build_vlm

    model = build_vlm().to(device)
    print(model)

    # Optimizer
    from schedulefree import AdamWScheduleFree
    optimizer = AdamWScheduleFree(model.parameters(), lr = 3e-4)

    # Data
    from dataset import get_coco_dataset#, get_pokemon_dataset
    from torch.utils.data import DataLoader

    #val_dataset = get_pokemon_dataset()
    #val_dataset = get_coco_dataset()
    train_dataset = get_coco_dataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size = 1)
    
    val_dataset = get_coco_dataset(mode = 'val')
    val_dataloader = DataLoader(val_dataset, batch_size = 1)

    print(train_dataset, val_dataset)
    n_epochs = 2
    
    train_model(model, n_epochs)
