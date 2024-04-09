"""
Train a vlm
"""


def train_model():
    
    model.train()
    optimizer.train()
    
    for _ in range(n_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            logits, loss = model.forward(batch)
            loss.backward()
            optimizer.step()
            
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

    model = build_vlm()
    print(model)

    # Optimizer
    from schedulefree import AdamWScheduleFree
    optimizer = AdamWScheduleFree(model.parameters(), lr = 3e-4)

    # Data
    from dataset import get_coco_dataset#, get_pokemon_dataset
    from torch.utils.data import DataLoader

    #val_dataset = get_pokemon_dataset()
    #val_dataset = get_coco_dataset()
    train_dataset = get_coco_dataset()
    
    train_dataloader = DataLoader(train_dataset, batch_size = 1)

    print(train_dataset, val_dataset)
    n_epochs = 2
    
    train_model()