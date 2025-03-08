from torch.utils.data.distributed import DistributedSampler

def load_data(train_dir, test_dir, batch_size=32, distributed=False):
    # Define transforms
    train_transform = get_train_transform()
    test_transform = get_test_transform()

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Create DistributedSampler if distributed training is enabled
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Shuffle only if no sampler is provided
        sampler=train_sampler,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, test_loader, train_sampler
