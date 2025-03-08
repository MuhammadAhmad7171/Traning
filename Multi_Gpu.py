import torch
import torch.distributed as dist
import torch.nn as nn

def setup_distributed_training(rank, world_size):
    """Initialize distributed training setup"""
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set the device based on the number of available GPUs
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(rank)  # Use the GPU corresponding to the rank
    else:
        torch.cuda.set_device(0)  # Use the only available GPU

def setup_multi_gpu(model):
    """Wrap model in DistributedDataParallel (DDP) if multiple GPUs are available"""
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    return model
