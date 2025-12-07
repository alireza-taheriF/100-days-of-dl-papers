import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from model import ResNet18, PlainNet18
from data_setup import get_data_loaders

def train(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple MPS (M2 Acceleration)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (This will be slow!)")

    print("📦 Loading Data...")
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)

    if args.model == 'resnet':
        print("🏗️ Building ResNet-18 (With Skip Connections)...")
        model = ResNet18(num_classes=10).to(device)
    else:
        print("🏗️ Building PlainNet-18 (NO Skip Connections)...")
        model = PlainNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()

        acc_train = 100. * correct_train / total_train
        acc_test = 100. * correct_test / total_test
        avg_train_loss = train_loss / len(train_loader)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"LR: {current_lr:.4f} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {acc_train:.2f}% | "
              f"Test Acc: {acc_test:.2f}%")

        if acc_test > best_acc:
            print(f"⭐ New Best Accuracy! ({best_acc:.2f}% -> {acc_test:.2f}%) Saving model...")
            best_acc = acc_test
            torch.save(model.state_dict(), 'resnet18_cifar10_best.pth')

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training Finished in {total_time:.1f} minutes.")
    print(f"🏆 Best Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'plain'], help='Model type')
    args = parser.parse_args()

    train(args)