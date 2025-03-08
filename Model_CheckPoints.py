import os  # Add this import
import torch

def save_checkpoint(model, optimizer, epoch, loss, is_best, checkpoint_dir):
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    filename = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, filename)
    
    if is_best:
        torch.save(checkpoint, f"{checkpoint_dir}/best_model.pth")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss
