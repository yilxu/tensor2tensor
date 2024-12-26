import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from transformer_engine import pytorch as te
from torch.utils.data import DataLoader
from custom_mnist_dataset import CustomMNISTDataset
import pandas as pd
import numpy as np

class Net(nn.Module):
    def __init__(self, use_te=False):
        super(Net, self).__init__()
        # CNN for 28x28 input
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        LinearClass = te.Linear if use_te else nn.Linear
        # After conv: 28x28 -> conv1(3x3): 26x26 -> conv2(3x3): 24x24 -> max_pool(2x2): 12x12
        # So output size after convs/pool: 64 * 12 * 12 = 9216
        self.fc1 = LinearClass(9216, 128)
        self.fc2 = LinearClass(128, 16)
        self.fc3 = nn.Linear(16, 1)  # single output for regression

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # single continuous value

def train(args, model, device, train_loader, optimizer, epoch, use_fp8, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with te.fp8_autocast(enabled=use_fp8):
            output = model(data)
        loss = criterion(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
            )
            if args.dry_run:
                break

def test(model, device, test_loader, use_fp8, criterion):
    model.eval()
    test_loss = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with te.fp8_autocast(enabled=use_fp8):
                output = model(data)
            loss = criterion(output, target.unsqueeze(1))
            test_loss += loss.item() * len(data)
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average MSE loss: {test_loss:.4f}\n")

    predictions = np.concatenate(predictions, axis=0).flatten()
    targets = np.concatenate(targets, axis=0)
    return predictions, targets

def main():
    parser = argparse.ArgumentParser(description="PyTorch Example with Transformer Engine (Regression)")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N")
    parser.add_argument("--epochs", type=int, default=5, metavar="N")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--use-te", action="store_true", default=False, help="Use Transformer Engine")
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Ubyte files must have 784 features for images (after padding/truncation)
    train_image_file = "train-images-idx3-ubyte"
    #lable is the return rate
    train_label_file = "train-labels-idx1-ubyte"
    test_image_file = "t10k-images-idx3-ubyte"
    test_label_file = "t10k-labels-idx1-ubyte"

    dataset1 = CustomMNISTDataset(train_image_file, train_label_file)
    dataset2 = CustomMNISTDataset(test_image_file, test_label_file)

    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    model = Net(use_te=args.use_te).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, use_fp8=False, criterion=criterion)
        predictions, targets = test(model, device, test_loader, use_fp8=False, criterion=criterion)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "msft_model_cnn_regression.pt")
        print("Model saved as msft_model_cnn_regression.pt")

    # Save predictions and targets to Excel
    df_results = pd.DataFrame({
        'Predictions': predictions,
        'Targets': targets
    })
    df_results.to_excel("predictions.xlsx", index=False)
    print("Predictions saved to predictions.xlsx")

if __name__ == "__main__":
    main()
