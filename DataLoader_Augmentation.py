import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_data(train_dir, test_dir, batch_size=32, distributed=False):
    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transform())
    test_dataset = datasets.ImageFolder(test_dir, transform=get_test_transform())

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=4, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=test_sampler)

    return train_loader, test_loader, train_sampler
