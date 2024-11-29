import multiprocessing
import os
import subprocess

# 定义训练参数
NUM_TASKS = 5  # 模拟任务数量（对应数据集分片数量）
DATA_BASE_PATH = "dataset"  # 数据集根路径
EPOCHS = 10
BATCH_SIZE = 64
LATENT_DIM = 100
IMG_SIZE = 28


def run_task(rank):
    """
    模拟单个任务的训练过程
    """
    # 动态设置每个任务加载的子数据集路径
    data_path = os.path.join(DATA_BASE_PATH, f"split_{rank}")
    if not os.path.exists(data_path):
        print(f"Data path for rank {rank} does not exist: {data_path}")
        return

    print(f"Starting task {rank} with data path: {data_path}...")
    command = [
        "python",
        "train.py",
        "--data_path", data_path,  # 传递子数据集路径
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--latent_dim", str(LATENT_DIM),
        "--img_size", str(IMG_SIZE),
        "--rank", str(rank),
    ]
    subprocess.run(command)
    print(f"Task {rank} finished.")


def main():
    """
    模拟多个任务的独立运行
    """
    print("Starting HTC simulation locally...")
    # 使用多进程模拟任务并行运行
    processes = []
    for rank in range(NUM_TASKS):
        p = multiprocessing.Process(target=run_task, args=(rank,))
        p.start()
        processes.append(p)

    # 等待所有任务完成
    for p in processes:
        p.join()

    print("All tasks completed.")


if __name__ == "__main__":
    main()
