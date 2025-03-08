import torch
import torch.nn as nn

def setup_multi_gpu(model):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    return model
