import os
import shutil
from math import ceil

# 输入参数
ORIGIN_DATASET = "origin_dataset"  # 原始数据集路径
OUTPUT_DIR = "dataset"  # 输出路径
NUM_SPLITS = 5  # 数据集分片数量


def split_dataset(origin_dataset, output_dir, num_splits):
    """
    将 origin_dataset 中的 real_images 和 fake_images 按输入数量均分到多个子数据集中
    并符合 ImageFolder 的结构
    """
    real_images_dir = os.path.join(origin_dataset, "real_images")
    fake_images_dir = os.path.join(origin_dataset, "fake_images")

    if not os.path.exists(real_images_dir) or not os.path.exists(fake_images_dir):
        print("Error: origin_dataset must contain 'real_images' and 'fake_images' directories.")
        return

    # 获取所有图片文件列表
    real_images = sorted(os.listdir(real_images_dir))
    fake_images = sorted(os.listdir(fake_images_dir))

    total_real = len(real_images)
    total_fake = len(fake_images)

    print(f"Total real images: {total_real}, Total fake images: {total_fake}")

    # 计算每个分片的图片数量
    real_per_split = ceil(total_real / num_splits)
    fake_per_split = ceil(total_fake / num_splits)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 开始分片
    for i in range(num_splits):
        split_dir = os.path.join(output_dir, f"split_{i}")
        os.makedirs(os.path.join(split_dir, "real_images", "real"), exist_ok=True)  # 创建类别子文件夹
        os.makedirs(os.path.join(split_dir, "fake_images", "fake"), exist_ok=True)  # 创建类别子文件夹

        # 分配 real_images
        real_start_idx = i * real_per_split
        real_end_idx = min(real_start_idx + real_per_split, total_real)
        for img in real_images[real_start_idx:real_end_idx]:
            shutil.copy(os.path.join(real_images_dir, img), os.path.join(split_dir, "real_images", "real", img))

        # 分配 fake_images
        fake_start_idx = i * fake_per_split
        fake_end_idx = min(fake_start_idx + fake_per_split, total_fake)
        for img in fake_images[fake_start_idx:fake_end_idx]:
            shutil.copy(os.path.join(fake_images_dir, img), os.path.join(split_dir, "fake_images", "fake", img))

        print(f"Split {i} created: {real_end_idx - real_start_idx} real images, {fake_end_idx - fake_start_idx} fake images")

    print(f"Dataset successfully split into {num_splits} parts.")


# 执行分割
split_dataset(ORIGIN_DATASET, OUTPUT_DIR, NUM_SPLITS)
