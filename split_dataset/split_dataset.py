import os
import shutil
from math import ceil
import random
from concurrent.futures import ThreadPoolExecutor


def copy_files(files, src_dir, dst_dir):
    """
    并行复制文件
    """
    for img in files:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))


def split_dataset(origin_dataset, output_dir, num_splits):
    real_images_dir = os.path.join(origin_dataset, "real_images")
    fake_images_dir = os.path.join(origin_dataset, "fake_images")

    if not os.path.exists(real_images_dir) or not os.path.exists(fake_images_dir):
        print("Error: origin_dataset must contain 'real_images' and 'fake_images' directories.")
        return

    # 获取所有图片文件列表并打乱顺序
    real_images = sorted(os.listdir(real_images_dir))
    fake_images = sorted(os.listdir(fake_images_dir))
    random.shuffle(real_images)
    random.shuffle(fake_images)

    total_real = len(real_images)
    total_fake = len(fake_images)
    print(f"Total real images: {total_real}, Total fake images: {total_fake}")

    # 计算每个分片的图片数量
    real_per_split = ceil(total_real / num_splits)
    fake_per_split = ceil(total_fake / num_splits)
    os.makedirs(output_dir, exist_ok=True)

    # 开始分片
    with ThreadPoolExecutor() as executor:
        for i in range(num_splits):
            split_dir = os.path.join(output_dir, f"split_{i}")
            os.makedirs(os.path.join(split_dir, "real"), exist_ok=True)
            os.makedirs(os.path.join(split_dir, "fake"), exist_ok=True)

            real_start_idx = i * real_per_split
            real_end_idx = min(real_start_idx + real_per_split, total_real)
            fake_start_idx = i * fake_per_split
            fake_end_idx = min(fake_start_idx + fake_per_split, total_fake)

            # 并行复制 real_images 和 fake_images
            executor.submit(
                copy_files, real_images[real_start_idx:real_end_idx], real_images_dir, os.path.join(split_dir, "real")
            )
            executor.submit(
                copy_files, fake_images[fake_start_idx:fake_end_idx], fake_images_dir, os.path.join(split_dir, "fake")
            )

            print(f"Split {i} created: {real_end_idx - real_start_idx} real images, {fake_end_idx - fake_start_idx}")

    print(f"Dataset successfully split into {num_splits} parts.")


# 输入参数
ORIGIN_DATASET = "origin_dataset"  # 原始数据集路径
OUTPUT_DIR = "dataset"  # 输出路径
NUM_SPLITS = 5  # 数据集分片数量
# 执行分割
split_dataset(ORIGIN_DATASET, OUTPUT_DIR, NUM_SPLITS)
