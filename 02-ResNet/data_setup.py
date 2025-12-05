import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def get_data_loaders(batch_size=128, num_workers=2):
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    print("⏳ Downloading CIFAR-10 dataset...")
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def denormalize(img):
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    img = img.clone()
    for t, m, s in zip(img, stats[0], stats[1]):
        t.mul_(s).add_(m)
    return img

if __name__ == "__main__":
    train_loader, _ = get_data_loaders(batch_size=4)
    
    images, labels = next(iter(train_loader))
    print(f"Batch Shape: {images.shape}")
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        img = denormalize(images[i])
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(classes[labels[i]])
        ax.axis('off')
    
    plt.show()
    print("✅ Data pipeline ready with Augmentation!")