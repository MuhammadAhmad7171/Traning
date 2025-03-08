import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.models as models
from torch.utils.data import DistributedSampler

from multigpu import setup_distributed_training, setup_multi_gpu
from modelcheckpionts import save_checkpoint
from dataloader_augmentation import load_data
from traininglogs import setup_logger

def main():
    # Distributed setup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_distributed_training(rank, world_size)
    
    # Logging
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    # Load dataset
    train_dir = "./data/train"
    test_dir = "./data/test"
    batch_size = 32
    train_loader, test_loader = load_data(train_dir, test_dir, batch_size)

    # Load model and wrap with DDP
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.to(rank)
    model = setup_multi_gpu(model)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Track best loss
    best_loss = float("inf")

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)  # Ensure correct shuffling across GPUs
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")

        # Checkpoint saving
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save last model
        save_checkpoint(model, optimizer, epoch, avg_loss, is_best=False, checkpoint_dir=checkpoint_dir)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, is_best=True, checkpoint_dir=checkpoint_dir)

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

