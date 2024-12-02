import numpy as np
from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor


def load_and_save_dataset(num):
    """
    加载数据集，进行变换，保存为 .npz 格式
    """
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
    print(f"Dataset split_{num} saved.")


def process_datasets_concurrently(num_splits):
    """
    使用多线程处理多个分片数据集
    """
    with ThreadPoolExecutor() as executor:
        executor.map(load_and_save_dataset, range(num_splits))


# 执行并行处理
if __name__ == "__main__":
    NUM_SPLITS = 5
    process_datasets_concurrently(NUM_SPLITS)
