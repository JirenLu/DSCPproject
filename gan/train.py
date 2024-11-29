import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from generator import Generator
from discriminator import Discriminator
import argparse
import os

def load_datasets(data_path, img_size):
    """
    加载 real_images 和 fake_images 两个子文件夹的数据集
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    real_data = ImageFolder(root=os.path.join(data_path, "real_images"), transform=transform)
    fake_data = ImageFolder(root=os.path.join(data_path, "fake_images"), transform=transform)
    #print(real_data, fake_data)
    return real_data, fake_data

def train(data_path, latent_dim, img_size, batch_size, epochs, rank):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Load datasets
    real_data, fake_data = load_datasets(data_path, img_size)
    real_loader = DataLoader(real_data, batch_size=batch_size, shuffle=True)
    fake_loader = DataLoader(fake_data, batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = Generator(latent_dim, img_size**2).to(device)
    discriminator = Discriminator(img_size**2).to(device)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        real_iter = iter(real_loader)
        fake_iter = iter(fake_loader)

        for _ in range(len(real_loader)):
            try:
                real_data, _ = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_data, _ = next(real_iter)

            real_data = real_data.view(real_data.size(0), -1).to(device)
            batch_size = real_data.size(0)

            # Generate fake data
            noise = torch.randn(batch_size, latent_dim).to(device)
            generated_data = generator(noise).detach()

            # Train Discriminator
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(real_data), real_labels)
            fake_loss = criterion(discriminator(generated_data), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            noise = torch.randn(batch_size, latent_dim).to(device)
            generated_data = generator(noise)
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(generated_data), real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Rank {rank} | Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Save models
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(generator.state_dict(), f"checkpoints/generator_rank_{rank}.pt")
    torch.save(discriminator.state_dict(), f"checkpoints/discriminator_rank_{rank}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent GAN Training with Split Dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset split (e.g., split_0)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent dimension")
    parser.add_argument("--img_size", type=int, default=28, help="Image size")
    parser.add_argument("--rank", type=int, required=True, help="Unique rank for this task")
    args = parser.parse_args()

    train(args.data_path, args.latent_dim, args.img_size, args.batch_size, args.epochs, args.rank)
