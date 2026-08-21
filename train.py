import os
import sys
import json
import random
import argparse
import subprocess
import time
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from models.faster_rcnn import FasterRCNN
from utils.dataset import ObjectDetectionDataset, IDX_TO_CLASS, detection_collate_fn
from utils.augmentations import DetectionTransforms


def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN Training Script from Scratch (Supports Multi-GPU DDP & TensorBoard)")
    parser.add_argument("--train_data", type=str, default="./public/annotations/train.json", help="Path to train annotation JSON")
    parser.add_argument("--val_data", type=str, default="./public/annotations/val.json", help="Path to val annotation JSON")
    parser.add_argument("--image_dir", type=str, default="./public/train/images", help="Path to train images directory")
    parser.add_argument("--val_image_dir", type=str, default="./public/val/images", help="Path to val images directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./models/", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./runs/", help="Directory to save TensorBoard logs")

    # Backbone architecture selection
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"], help="Backbone architecture")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU (default: 8)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"], help="Optimizer type (default: sgd)")
    parser.add_argument("--lr", type=float, default=0.01, help="Base learning rate for heads/RPN (default: 0.01 for SGD, 1e-4 for AdamW)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1, help="LR multiplier for backbone fine-tuning (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num workers")
    parser.add_argument("--img_size", type=int, default=640, help="Target image square size for training")
    parser.add_argument("--fc_dim", type=int, default=1024, help="FC dimension in Fast R-CNN Head (default: 1024)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability in Fast R-CNN Head (default: 0.3)")

    # Advanced training options
    parser.add_argument("--use_expand_crop", action="store_true", default=False, help="Enable random expand + crop data augmentation (default: False)")
    parser.add_argument("--use_mosaic", action="store_true", default=False, help="Enable selective mosaic 4-image data augmentation (default: False)")
    parser.add_argument("--mosaic_prob", type=float, default=0.3, help="Probability for mosaic augmentation when enabled (default: 0.3)")
    parser.add_argument("--warmup_iters", type=int, default=500, help="Number of warmup iterations for LR")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable TensorBoard logging")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DDP Helpers
# ---------------------------------------------------------------------------

def is_main_process(rank):
    return rank == 0


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return is_distributed, rank, local_rank, world_size, device


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Learning Rate Warmup Scheduler (Multi-Parameter Groups)
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_iters` steps, then cosine annealing.
    Supports differential learning rates across multiple parameter groups.
    """

    def __init__(self, optimizer, warmup_iters, total_iters, base_lrs=None):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        if base_lrs is None:
            self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        else:
            self.base_lrs = base_lrs
        self.current_iter = 0

    def step(self):
        self.current_iter += 1
        if self.current_iter <= self.warmup_iters:
            factor = self.current_iter / max(self.warmup_iters, 1)
        else:
            progress = (self.current_iter - self.warmup_iters) / max(self.total_iters - self.warmup_iters, 1)
            import math
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * factor


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, scheduler_iter, grad_clip, rank=0, writer=None):
    model.train()
    running_loss = 0.0
    running_loss_details = {}
    start_time = time.time()

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            loss_dict = model(images, targets)
            loss = loss_dict["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        if scheduler_iter is not None:
            scheduler_iter.step()

        running_loss += loss.item()
        for k, v in loss_dict.items():
            if k != "loss":
                running_loss_details[k] = running_loss_details.get(k, 0.0) + v.item()

        head_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]
        backbone_lr = optimizer.param_groups[0]["lr"]

        # Log per-step metrics to TensorBoard
        if writer is not None and is_main_process(rank):
            global_step = epoch * len(dataloader) + step
            writer.add_scalar("Train_Step/Loss", loss.item(), global_step)
            for k, v in loss_dict.items():
                if k != "loss":
                    writer.add_scalar(f"Train_Step/{k}", v.item(), global_step)
            writer.add_scalar("Train_Step/LR_Head", head_lr, global_step)
            writer.add_scalar("Train_Step/LR_Backbone", backbone_lr, global_step)

        if is_main_process(rank) and ((step + 1) % 20 == 0 or (step + 1) == len(dataloader)):
            elapsed = time.time() - start_time
            detail_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items() if k != "loss"])
            print(
                f"  Epoch [{epoch+1}] Step [{step+1}/{len(dataloader)}] "
                f"Total Loss: {loss.item():.4f} ({detail_str}) "
                f"LR: {head_lr:.6f} Time: {elapsed:.1f}s"
            )

    epoch_loss = running_loss / len(dataloader)
    epoch_details = {k: v / len(dataloader) for k, v in running_loss_details.items()}
    return epoch_loss, epoch_details


# ---------------------------------------------------------------------------
# Validation — compute val loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_loss(model, dataloader, device):
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.train()  # Keep in train mode to compute val loss dict
    val_loss = 0.0
    val_details = {}

    for images, targets in dataloader:
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            loss_dict = raw_model(images, targets)
        val_loss += loss_dict["loss"].item()
        for k, v in loss_dict.items():
            if k != "loss":
                val_details[k] = val_details.get(k, 0.0) + v.item()

    n = max(len(dataloader), 1)
    val_details = {k: v / n for k, v in val_details.items()}
    return val_loss / n, val_details


# ---------------------------------------------------------------------------
# Validation — compute mAP
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_map(model, dataloader, device, val_ann_path, img_size, output_dir):
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    predictions = []
    for images, targets in dataloader:
        images = images.to(device)
        batch_results = raw_model(images)

        for i, det in enumerate(batch_results):
            img_id = targets[i]["image_id"]
            orig_h, orig_w = targets[i]["orig_shape"]

            boxes = det["boxes"].cpu()
            scores = det["scores"].cpu()
            labels = det["labels"].cpu()

            boxes_list = []
            if len(boxes) > 0:
                scale_x = orig_w / float(img_size)
                scale_y = orig_h / float(img_size)
                boxes[:, 0] = (boxes[:, 0] * scale_x).clamp(min=0, max=orig_w)
                boxes[:, 1] = (boxes[:, 1] * scale_y).clamp(min=0, max=orig_h)
                boxes[:, 2] = (boxes[:, 2] * scale_x).clamp(min=0, max=orig_w)
                boxes[:, 3] = (boxes[:, 3] * scale_y).clamp(min=0, max=orig_h)

                for box, score, label_idx in zip(boxes, scores, labels):
                    cls_name = IDX_TO_CLASS[label_idx.item()]
                    xmin, ymin, xmax, ymax = box.tolist()
                    if xmax <= xmin + 0.1 or ymax <= ymin + 0.1:
                        continue
                    boxes_list.append({
                        "class": cls_name,
                        "confidence": round(float(score.item()), 4),
                        "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                    })

            predictions.append({"image_id": img_id, "boxes": boxes_list})

    pred_path = os.path.join(output_dir, "_val_predictions_tmp.json")
    score_path = os.path.join(output_dir, "_val_score_tmp.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    eval_script = os.path.join("public", "tools", "evaluate_predictions.py")
    if not os.path.exists(eval_script):
        eval_script = os.path.join(".", "public", "tools", "evaluate_predictions.py")

    if os.path.exists(eval_script):
        try:
            result = subprocess.run(
                [sys.executable, eval_script,
                 "--ground_truth", val_ann_path,
                 "--predictions", pred_path,
                 "--output", score_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and os.path.exists(score_path):
                with open(score_path, "r") as f:
                    score_data = json.load(f)
                mAP = score_data.get("mAP@0.5", score_data.get("mAP", 0.0))
                return float(mAP)
        except Exception as e:
            print(f"  [WARN] Could not run evaluate_predictions.py: {e}")

    return None


# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    is_distributed, rank, local_rank, world_size, device = setup_distributed()

    if is_main_process(rank):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best.pth")

    # Initialize TensorBoard SummaryWriter on main process
    writer = None
    if is_main_process(rank) and not args.no_tensorboard:
        if SummaryWriter is not None:
            run_name = time.strftime(f"faster_rcnn_{args.backbone}_%Y%m%d_%H%M%S")
            log_dir = os.path.join(args.log_dir, run_name)
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging initialized at: {log_dir}")
        else:
            print("[INFO] tensorboard package not found, skipping TensorBoard logging.")

    if is_main_process(rank):
        print(f"Device: {device} | Distributed: {is_distributed} (World Size: {world_size})")

    # Build Data Pipelines
    train_transforms = DetectionTransforms(
        target_size=(args.img_size, args.img_size),
        is_train=True,
        multi_scale=True,
        use_expand_crop=args.use_expand_crop,
    )
    val_transforms = DetectionTransforms(
        target_size=(args.img_size, args.img_size), is_train=False, multi_scale=False
    )

    train_dataset = ObjectDetectionDataset(
        args.train_data,
        args.image_dir,
        transforms=train_transforms,
        use_mosaic=args.use_mosaic,
        mosaic_prob=args.mosaic_prob,
    )
    val_dataset = ObjectDetectionDataset(
        args.val_data,
        args.val_image_dir,
        transforms=val_transforms,
        use_mosaic=False,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    if is_main_process(rank):
        print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Initialize Faster R-CNN Model
    model = FasterRCNN(
        num_classes=len(train_dataset.classes),
        backbone_name=args.backbone,
        fc_dim=args.fc_dim,
        dropout_p=args.dropout,
        pretrained=True,
    ).to(device)

    # Wrap model with DDP if multi-GPU
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Differential Learning Rates: Backbone gets 10x smaller LR for safe fine-tuning
    raw_model = model.module if is_distributed else model
    backbone_params = list(raw_model.backbone.parameters())
    head_params = [p for n, p in raw_model.named_parameters() if not n.startswith("backbone")]

    effective_base_lr = args.lr * world_size
    effective_backbone_lr = effective_base_lr * args.backbone_lr_ratio

    param_groups = [
        {"params": backbone_params, "lr": effective_backbone_lr},
        {"params": head_params, "lr": effective_base_lr},
    ]

    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay,
        )

    total_iters = args.epochs * len(train_loader)
    scheduler_iter = WarmupCosineScheduler(
        optimizer,
        warmup_iters=args.warmup_iters,
        total_iters=total_iters,
        base_lrs=[effective_backbone_lr, effective_base_lr],
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_metric = -1.0
    best_val_loss = float("inf")
    use_map_tracking = True

    if is_main_process(rank):
        print("=" * 70)
        print(f"Starting training pipeline (FASTER R-CNN)...")
        print(f"  Backbone: {args.backbone}")
        print(f"  GPUs: {world_size}, Batch per GPU: {args.batch_size} (Total Batch: {args.batch_size * world_size})")
        print(f"  Epochs: {args.epochs}")
        print(f"  Heads Base LR: {effective_base_lr:.6f} | Backbone LR: {effective_backbone_lr:.6f}")
        print(f"  Warmup iters: {args.warmup_iters}, Grad clip: {args.grad_clip}")
        print(f"  Multi-scale training: ON, Image size: {args.img_size}")
        print("=" * 70)

    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # Disable mosaic augmentation in the last 5 epochs for clean fine-tuning
        if args.use_mosaic and epoch >= args.epochs - 5:
            train_dataset.use_mosaic = False
            if is_main_process(rank) and epoch == args.epochs - 5:
                print("Mosaic augmentation disabled for final 5 epochs (clean fine-tuning).")

        if is_main_process(rank):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss, train_details = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            scheduler_iter, args.grad_clip, rank=rank, writer=writer,
        )

        # Validation & checkpointing only on main process
        if is_main_process(rank):
            val_loss, val_details = validate_loss(model, val_loader, device)

            detail_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_details.items()])
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ({detail_str})")

            val_map = evaluate_map(
                model, val_loader, device, args.val_data, args.img_size, args.checkpoint_dir
            )

            # Log per-epoch metrics to TensorBoard
            if writer is not None:
                writer.add_scalar("Epoch/Train_Loss", train_loss, epoch + 1)
                writer.add_scalar("Epoch/Val_Loss", val_loss, epoch + 1)
                for k, v in train_details.items():
                    writer.add_scalar(f"Epoch/Train_{k}", v, epoch + 1)
                for k, v in val_details.items():
                    writer.add_scalar(f"Epoch/Val_{k}", v, epoch + 1)
                if val_map is not None:
                    writer.add_scalar("Epoch/Val_mAP50", val_map, epoch + 1)

            should_save = False

            if val_map is not None:
                print(f"  Val mAP@0.5: {val_map:.4f}")
                if val_map > best_metric:
                    best_metric = val_map
                    should_save = True
                    print(f"  ★ New best mAP@0.5: {val_map:.4f}")
            else:
                if use_map_tracking:
                    print("  [INFO] mAP evaluation unavailable, falling back to val loss tracking.")
                    use_map_tracking = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    should_save = True
                    print(f"  ★ New best val loss: {val_loss:.4f}")

            if should_save:
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_name": "faster_rcnn",
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": val_loss,
                        "best_map": val_map if val_map is not None else -1,
                        "classes": train_dataset.classes,
                        "args": vars(args),
                    },
                    best_ckpt_path,
                )
                print(f"  → Saved best checkpoint to {best_ckpt_path}")

        # Synchronize ranks across epochs safely
        if is_distributed:
            dist.barrier()

    if is_main_process(rank):
        print("=" * 70)
        print(f"Training completed! Best checkpoint: {best_ckpt_path}")
        if best_metric > 0:
            print(f"Best mAP@0.5: {best_metric:.4f}")
        print("=" * 70)

        if writer is not None:
            writer.close()
            print(f"📊 TensorBoard log saved to {writer.log_dir}")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
