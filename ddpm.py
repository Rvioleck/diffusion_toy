import argparse
import os
import sys
import torch
from trainer import train_model
from inference import generate_and_show_inference_images
from utils import show_first_batch
from dataset import get_data_loader


def main(args):
    # 参数解析
    parser = argparse.ArgumentParser(description="DDPM Training & Inference Script")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps for DDPM training/inference.")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory to store generated GIFs.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to store model checkpoints.")
    parser.add_argument("--model-path", help="Path to the trained model for inference")
    args = parser.parse_args(args)

    # 确保目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        # 加载数据
        loader = get_data_loader(args.dataset, args.batch_size)
        # 显示初始数据样本
        # show_first_batch(loader)
        # 训练模型
        train_model(loader, args.epochs, args.learning_rate, args.checkpoint_dir, args.dataset, device, args.steps)
        print(f"Model trained and checkpoint saved.")
    elif args.infer:
        # 推理
        if args.model_path is None:
            print("Error: Model path is required for inference.")
            return
        model_path = args.model_path
        generate_and_show_inference_images(model_path, args.log_dir, args.dataset, device, args.steps)
    else:
        print("No action specified. Use --train or --infer to start.")


if __name__ == "__main__":
    main(sys.argv[1:])
