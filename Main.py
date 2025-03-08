import torch
import torch.optim as optim
import torch.distributed as dist
import os
import argparse
from torchvision import models
from Training_logs import setup_logger
from Model_CheckPoints import save_checkpoint, load_checkpoint
from Multi_Gpu import setup_distributed_training
from DataLoader_Augmentation import load_data

# Argument Parser for Configurations
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--world_size', type=int, default=2, help='Number of nodes for distributed training')
    return parser.parse_args()

# Compute Top-1 and Top-5 Accuracy
def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True) * 100.0 / batch_size for k in topk]

# Main Training Function
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use GPUs
    
    # Get rank and world_size from environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Initialize Distributed Training
    setup_distributed_training(rank, world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0:
        logger = setup_logger("./logs")
    
    # Model Setup
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 5)
    model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Data Loaders
    train_loader, test_loader = load_data("/kaggle/input/alzheimer-5-class/Alzheimer 5 classes/train", 
                                          "/kaggle/input/alzheimer-5-class/Alzheimer 5 classes/test", 
                                          args.batch_size)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Load Checkpoint if Resuming Training
    start_epoch = 0
    if args.resume:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
    
    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, top1_acc_total, top5_acc_total, total_samples = 0.0, 0.0, 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(rank), labels.cuda(rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            top1, top5 = accuracy(outputs, labels)
            top1_acc_total += top1.item()
            top5_acc_total += top5.item()
            total_samples += 1
        
        avg_top1_acc = top1_acc_total / total_samples
        avg_top5_acc = top5_acc_total / total_samples
        
        if rank == 0:
            logger.info(f"Epoch {epoch}, Loss: {running_loss/len(train_loader)}, Top-1 Acc: {avg_top1_acc:.2f}%, Top-5 Acc: {avg_top5_acc:.2f}%")
        
        # Save checkpoint after each epoch
        if rank == 0:
            save_checkpoint(model, optimizer, epoch, running_loss, is_best=False, checkpoint_dir="./checkpoints")
    
    # Save final model
    if rank == 0:
        save_checkpoint(model, optimizer, epoch, running_loss, is_best=True, checkpoint_dir="./checkpoints")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
