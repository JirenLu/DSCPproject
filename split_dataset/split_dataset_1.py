import numpy as np
from torchvision import datasets, transforms


def load_and_save_datasets(num):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((200, 200)),  # Resize to 200x200
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=f"dataset/split_{num}", transform=transform)

    # Collect data and labels
    all_images = []
    all_labels = []
    for img, label in dataset:
        all_images.append(img.numpy())  # Convert tensors to numpy arrays
        all_labels.append(label)

    # Convert lists to arrays
    all_images = np.stack(all_images, axis=0)  # Stack into (N, C, H, W)
    all_labels = np.array(all_labels)  # Convert labels to NumPy array

    # Save to .npz
    np.savez_compressed(f"dataset/split_{num}/split_{num}", images=all_images, labels=all_labels)
    print(f"Dataset saved")


for i in range(5):
    load_and_save_datasets(i)