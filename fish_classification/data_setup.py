import os
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()
def create_data(
        image_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    data = datasets.ImageFolder(image_dir,transform=transform)

    class_names = data.classes

    train_size = int(len(data)*0.8)
    test_size = len(data)-train_size
    train_data,test_data = random_split(data,[train_size,test_size])

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names