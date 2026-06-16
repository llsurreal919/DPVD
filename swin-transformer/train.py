import os
import argparse
import torch
import torch.optim as optim
import pandas as pd
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from rich.console import Console
from rich.table import Table
from rich import box
from tqdm import tqdm
import sys  # 如果你想定向 tqdm 输出
from my_dataset import MyDataSet
from model import swin_base_patch4_window7_224_in22k as create_model
from utils import read_split_data, evaluate, read_fixed_split_data


def main(args):
    console = Console()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = "train5"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "weights"), exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    train_log_file = open(os.path.join(output_dir, "train_log.txt"), "w")
    val_log_file = open(os.path.join(output_dir, "val_log.txt"), "w")
    batch_log_file = open(os.path.join(output_dir, "batch_log.txt"), "w")

    console.rule("[bold cyan]🚀 Dataset Loading")
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.data_path)
    console.print(f"[green]Total images:[/] {len(train_images_path) + len(val_images_path)}")
    console.print(f"[green]Training:[/] {len(train_images_path)}   [green]Validation:[/] {len(val_images_path)}")

    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = 8
    console.print(f"[green]Dataloader workers:[/] {nw}")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    console.rule("[bold cyan]🧠 Model Setup")
    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights:
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        try:
            weights_dict = torch.load(args.weights, map_location=device, weights_only=True)["model"]
        except TypeError:
            weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)
        console.print("[yellow]Pretrained weights loaded (head ignored).[/]")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                console.print(f"[cyan]Training:[/] {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3,
        threshold=0.0001, threshold_mode='rel', cooldown=0,
        min_lr=0, eps=1e-12)

    scaler = GradScaler(device='cuda')
    loss_function = torch.nn.CrossEntropyLoss()
    metrics = []
    console.rule("[bold green]📊 Training Start")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, file=sys.stdout, desc=f"[Epoch {epoch + 1}]")

        for step, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = correct / total
            progress_bar.set_postfix(loss=loss.item(), acc=acc)
            batch_log_file.write(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item():.4f}, Acc: {acc:.4f}\n")

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_log_file.write(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}\n")

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        val_log_file.write(f"Epoch {epoch + 1}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n")

        lr_current = optimizer.param_groups[0]["lr"]

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", lr_current, epoch)

        torch.save(model.state_dict(), os.path.join(output_dir, "weights", f"model-{epoch}.pth"))

        metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr_current
        })

        scheduler.step(val_loss)

        table = Table(title=f"Epoch {epoch + 1}/{args.epochs} Summary", box=box.SIMPLE_HEAVY)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")
        table.add_row("Train Loss", f"{train_loss:.4f}")
        table.add_row("Train Acc", f"{train_acc:.2%}")
        table.add_row("Val Loss", f"{val_loss:.4f}")
        table.add_row("Val Acc", f"{val_acc:.2%}")
        table.add_row("Learning Rate", f"{lr_current:.6f}")
        console.print(table)

    df = pd.DataFrame(metrics)
    df.to_excel(os.path.join(output_dir, "metrics_log.xlsx"), index=False)
    console.rule(f"[bold green]✅ Training Complete - Outputs saved to {output_dir}")

    tb_writer.close()
    train_log_file.close()
    val_log_file.close()
    batch_log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str, default="E:/swin-transformer/newn5_dataset_spilt/train")
    parser.add_argument('--weights', type=str, default='./swin_base_patch4_window7_224_22k.pth')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    opt = parser.parse_args()
    main(opt)
