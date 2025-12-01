import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model import LeNet5
from data_setup import get_data_loaders

# Terminal input settings
parser = argparse.ArgumentParser(description='Train LeNet-5 on MNIST')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
args = parser.parse_args()

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load data and model
train_loader, test_loader = get_data_loaders(args.batch_size)
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

# Training loop
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{args.epochs} | Loss: {running_loss/len(train_loader):.4f}")

# Save the final model
torch.save(model.state_dict(), "lenet5_final.pth")
print("✅ Training finished & model saved!")