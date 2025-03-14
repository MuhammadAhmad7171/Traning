import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.models as models
from torch.utils.data import DistributedSampler

from Multi_Gpu import setup_distributed_training, setup_multi_gpu
from Model_CheckPoints import save_checkpoint
from DataLoader_Augmentation import load_data
from Training_logs import setup_logger

def compute_accuracy(outputs, labels, top_k=(1,)):
    """Compute top-k accuracy"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = labels.size(0)
        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        
        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    # Distributed setup
    available_gpus = torch.cuda.device_count()
    world_size = available_gpus  # Use all available GPUs
    rank = int(os.environ.get("RANK", 0))  # Default to 0 if RANK is not set
    
    # If only 1 GPU is available, disable distributed training
    if world_size == 1:
        print("Running in single-GPU mode (no distributed training)")
        torch.cuda.set_device(0)
    else:
        setup_distributed_training(rank, world_size)
    
    # Logging
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    # Load dataset
    train_dir = "/kaggle/input/alzheimer-5-class/Alzheimer 5 classes/train"
    test_dir = "/kaggle/input/alzheimer-5-class/Alzheimer 5 classes/test"  # Fix: Use the correct test directory
    batch_size = 32
    
    # Load data with distributed sampler
    train_loader, test_loader, train_sampler = load_data(train_dir, test_dir, batch_size, distributed=(world_size > 1))

    # Model initialization
    model = models.vit_b_16(weights=None)  # Initialize model without weights

    # Rank 0 downloads the weights and saves them to a file
    if rank == 0:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        torch.save(model.state_dict(), "vit_b_16_weights.pth")
    
    # Synchronize all processes to ensure rank 0 finishes downloading
    if world_size > 1:
        dist.barrier()

    # All ranks load the weights from the file
    if rank != 0:
        model.load_state_dict(torch.load("vit_b_16_weights.pth", map_location=f"cuda:{rank}"))

    # Move model to the correct device and wrap with DDP
    model.to(rank)
    model = setup_multi_gpu(model)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Track best validation loss
    best_val_loss = float("inf")

    # Training loop
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        
        # Set epoch for DistributedSampler
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        running_loss, correct_top1, correct_top5, total_samples = 0.0, 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update training loss & accuracy
            running_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            top1, top5 = compute_accuracy(outputs, labels, top_k=(1, 5))
            correct_top1 += top1.item() * images.size(0)
            correct_top5 += top5.item() * images.size(0)

        # Calculate epoch metrics
        train_loss = running_loss / total_samples
        train_acc1 = correct_top1 / total_samples
        train_acc5 = correct_top5 / total_samples

        # Validation phase
        model.eval()
        val_loss, val_correct_top1, val_correct_top5, val_samples = 0.0, 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(rank), labels.to(rank)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_samples += labels.size(0)

                top1, top5 = compute_accuracy(outputs, labels, top_k=(1, 5))
                val_correct_top1 += top1.item() * images.size(0)
                val_correct_top5 += top5.item() * images.size(0)

        # Compute validation metrics
        val_loss /= val_samples
        val_acc1 = val_correct_top1 / val_samples
        val_acc5 = val_correct_top5 / val_samples

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc1:.2f}% (Top-1) {train_acc5:.2f}% (Top-5) "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc1:.2f}% (Top-1) {val_acc5:.2f}% (Top-5)"
        )

        # Checkpoint saving
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, is_best=False, checkpoint_dir=checkpoint_dir)

        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, is_best=True, checkpoint_dir=checkpoint_dir)

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
