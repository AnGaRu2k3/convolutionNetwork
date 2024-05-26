from argparse import ArgumentParser
import os
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from net import ANN, CNN  # Update these imports based on your new models
from dataset import get_dataset  # Assume this is updated to handle Caltech101 and Caltech256
from optim import CosineSchedule
from engine import train_one_epoch, valid_one_epoch

import yaml

def create_args():
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--eval-log-dir", default="", type=str)
    parser.add_argument("--model", required=True, type=str, help="Path to model config file")
    parser.add_argument("--model-type", choices=["ann", "cnn"], required=True, help="Type of model to use (ann or cnn)")
    parser.add_argument("--mlp-dropout-rate", default=0, type=float)
    parser.add_argument("--conv-dropout-rate", default=0, type=float)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST", "Caltech101", "Caltech256"], required=True)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--base-lr", default=0.005, type=float)
    parser.add_argument("--target-lr", default=0.00001, type=float)
    parser.add_argument("--warmup-epochs", default=5, type=int)
    parser.add_argument("--max-epochs", default=70, type=int)
    parser.add_argument("--fig-dir", default="", type=str)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--weight-decay", default=0.001, type=float)
    parser.add_argument("--save-path", default="", type=str)
    parser.add_argument("--save-freq", default=5, type=int)
    parser.add_argument("--save-best-path", default="", type=str)
    parser.add_argument("--load-ckpt", default="", type=str)
    return parser.parse_args()

def main(args):
    # Build dataset based on arguments specified by user
    num_classes, train_dataset, valid_dataset = get_dataset(args.dataset)

    print("=" * os.get_terminal_size().columns)
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(valid_dataset)} samples")
    print("=" * os.get_terminal_size().columns)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(224),
        ]),
    }

    device = torch.device(args.device)

    # Build neural network using arguments passed by user
    with open(args.model, "r") as f:
        model_config = yaml.safe_load(f)
        
    if args.model_type == "ann":
        model = ANN(num_classes, args.mlp_dropout_rate)
    elif args.model_type == "cnn":
        model = CNN(3, num_classes, model_config["model"]["net"], model_config["model"]["mlp"], args.mlp_dropout_rate, args.conv_dropout_rate, model_config["model"]["max_pool_stride"])
    
    print("=" * os.get_terminal_size().columns)
    summary(model, (3, 224, 224))
    print("=" * os.get_terminal_size().columns)
    model.to(device)
    
    if len(args.load_ckpt):
        state_dict = torch.load(args.load_ckpt, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {args.load_ckpt}")

    loss_fn = nn.CrossEntropyLoss()
    if args.train:
        train_loss_values = []
        valid_loss_values = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
        scheduler = CosineSchedule(optimizer, base_lr=args.base_lr, target_lr=args.target_lr, max_steps=args.max_epochs, warmup_steps=args.warmup_epochs)

        if len(args.save_path) > 0:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        if len(args.save_best_path) > 0:
            os.makedirs(os.path.dirname(args.save_best_path), exist_ok=True)
        if len(args.fig_dir) > 0:
            os.makedirs(args.fig_dir, exist_ok=True)
        
        min_loss = math.inf
        not_better = 0
        
        for e in range(1, args.epochs + 1):
            train_loss = train_one_epoch(e, model, optimizer, loss_fn, train_dataloader, device, transform=image_transforms["train"], lr_scheduler=scheduler)
            acc, valid_loss, _, _ = valid_one_epoch(model, loss_fn, None, valid_dataloader, device, transform=image_transforms["valid"])
            
            train_loss_values.append(train_loss)
            valid_loss_values.append(valid_loss)

            if len(args.save_best_path) > 0 and valid_loss < min_loss:
                torch.save(model.state_dict(), args.save_best_path)
                print(f"Saved model to {args.save_best_path}")
            if min_loss <= valid_loss:
                not_better += 1
            else:
                not_better = 0

            min_loss = min(min_loss, valid_loss)
            if not_better == args.patience:
                print(f"Stopping because of exceeding patience")
                if len(args.save_path) > 0:
                    torch.save(model.state_dict(), args.save_path)
                    print(f"Saved model to {args.save_path}")
                break

            if len(args.save_path) > 0 and (e % args.save_freq == 0 or e == args.epochs):
                torch.save(model.state_dict(), args.save_path)
                print(f"Saved model to {args.save_path}")
        
        if len(args.fig_dir) > 0:
            x = np.arange(1, len(train_loss_values) + 1, dtype=np.int32)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(x, np.array(train_loss_values), color="g", label="train")
            plt.plot(x, np.array(valid_loss_values), color="r", label="valid")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(args.fig_dir + "/" + args.save_path.split("/")[-1].split(".")[-2] + ".pdf")
    else:
        acc, loss, preds, labels = valid_one_epoch(model, loss_fn, train_dataloader, valid_dataloader, device, transform=image_transforms["valid"])
        if len(args.eval_log_dir) > 0:
            os.makedirs(os.path.dirname(args.eval_log_dir), exist_ok=True)
            with open(args.eval_log_dir, "w") as f:
                json.dump({"preds": preds, "labels": labels}, f)

if __name__ == "__main__":
    args = create_args()
    main(args)
