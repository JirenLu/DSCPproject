# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse


# 数据加载函数
def load_datasets(data_path, img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# 模型定义
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, img_size):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (img_size // 4) * (img_size // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# 训练函数
def train(data_path, img_size, batch_size, epochs, lr, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    dataloader = load_datasets(data_path, img_size, batch_size)

    # 初始化模型
    model = SimpleCNN(num_classes, img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), rf"checkpoints/cnn_model_{data_path[-1]}.pth")
    # print(f"Model saved to {os.path.join(checkpoint_folder, checkpoint_file)}")


# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple CNN on a dataset")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset (with subfolders for classes)")
    parser.add_argument("--img_size", type=int, default=64, help="Image size (default: 64x64)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes in the dataset")
    # parser.add_argument("--save_path", type=str, default="./checkpoints", help="Path to save the trained model")
    args = parser.parse_args()

    train(args.data_path, args.img_size, args.batch_size, args.epochs, args.lr, args.num_classes)
