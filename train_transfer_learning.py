"""
Transfer learning 3 model (ResNet-18, ResNet-50, MobileNetV3-Small) pada dataset balok.
Setelah training, dibuat chart perbandingan ketiga model.

Struktur dataset yang diharapkan:
    dataset/
      train/<kelas>/*.jpg
      val/<kelas>/*.jpg
      test/<kelas>/*.jpg

Cara pakai:
    pip install torch torchvision matplotlib scikit-learn
    python train_transfer_learning.py --data dataset --epochs 15

Hanya membuat ulang chart dari hasil training sebelumnya:
    python train_transfer_learning.py --plot-only
"""
import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MODEL_NAMES = ["ResNet-18", "ResNet-50", "MobileNetV3-Small"]
COLORS = {"ResNet-18": "#2a9d8f", "ResNet-50": "#e76f51", "MobileNetV3-Small": "#264653"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_loaders(data_dir, batch_size, img_size, workers):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # statistik ImageNet
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    data_dir = Path(data_dir)
    sets = {
        "train": datasets.ImageFolder(data_dir / "train", train_tf),
        "val": datasets.ImageFolder(data_dir / "val", eval_tf),
        "test": datasets.ImageFolder(data_dir / "test", eval_tf),
    }
    pin = torch.cuda.is_available()
    loaders = {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"),
                      num_workers=workers, pin_memory=pin)
        for k, v in sets.items()
    }
    return loaders, sets["train"].classes


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(name, num_classes):
    """Load bobot pretrained ImageNet lalu ganti layer klasifikasi terakhir."""
    if name == "ResNet-18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    elif name == "ResNet-50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    elif name == "MobileNetV3-Small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        head = m.classifier
    else:
        raise ValueError(name)
    return m, head


def set_backbone_trainable(model, head, trainable):
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head.parameters():  # head selalu dilatih
        p.requires_grad = True


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ----------------------------------------------------------------------------
# Training / evaluasi
# ----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def measure_cpu_latency(model, img_size, runs=50):
    """Waktu inferensi 1 gambar di CPU (ms) -- relevan untuk Raspberry Pi / edge."""
    model = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(10):  # warm-up
        model(x)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    return (time.perf_counter() - t0) / runs * 1000


def train_one_model(name, loaders, classes, args, device, out_dir):
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
    model, head = build_model(name, len(classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, best_epoch = 0.0, None, 0
    optimizer, scheduler = None, None
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Fase 1: backbone dibekukan, hanya head yang dilatih (feature extraction)
        # Fase 2: seluruh layer dibuka dengan LR kecil (fine-tuning)
        if epoch == 1:
            set_backbone_trainable(model, head, False)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if epoch == args.warmup + 1:
            set_backbone_trainable(model, head, True)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.warmup))

        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        if scheduler is not None and epoch > args.warmup:
            scheduler.step()

        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

        fase = "head-only" if epoch <= args.warmup else "fine-tune"
        print(f"Epoch {epoch:02d}/{args.epochs} [{fase:9s}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

    train_time = time.time() - t_start

    # Evaluasi test set memakai bobot terbaik (berdasarkan val acc)
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "classes": classes}, out_dir / f"{name}_best.pt")
    preds, labels = predict_all(model, loaders["test"], device)
    test_acc = float((preds == labels).mean())

    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    result = {
        "name": name,
        "params_m": count_params(model),
        "size_mb": (out_dir / f"{name}_best.pt").stat().st_size / 1e6,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "train_time_s": train_time,
        "cpu_latency_ms": measure_cpu_latency(model, args.img_size),
        "history": history,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
    print(f"-> Test acc {test_acc:.3f} | waktu training {train_time:.0f}s | "
          f"latensi CPU {result['cpu_latency_ms']:.1f} ms/gambar")
    return result


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def plot_curves(results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    panels = [("train_loss", "Training Loss"), ("val_loss", "Validation Loss"),
              ("val_acc", "Validation Accuracy")]
    for ax, (key, title) in zip(axes, panels):
        for r in results:
            ep = range(1, len(r["history"][key]) + 1)
            ax.plot(ep, r["history"][key], marker="o", ms=3, lw=2,
                    label=r["name"], color=COLORS[r["name"]])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[2].set_ylim(top=1.02)
    axes[0].legend()
    fig.suptitle("Kurva Training - Perbandingan 3 Model Transfer Learning", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_kurva_training.png", dpi=150)
    plt.close(fig)


def plot_bars(results, out_dir):
    names = [r["name"] for r in results]
    cols = [COLORS[n] for n in names]
    metrics = [
        ("Akurasi Test (%)", [r["test_acc"] * 100 for r in results], "{:.1f}"),
        ("Jumlah Parameter (juta)", [r["params_m"] for r in results], "{:.1f}"),
        ("Latensi Inferensi CPU (ms/gambar)", [r["cpu_latency_ms"] for r in results], "{:.1f}"),
        ("Waktu Training (detik)", [r["train_time_s"] for r in results], "{:.0f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, vals, fmt) in zip(axes.ravel(), metrics):
        bars = ax.bar(names, vals, color=cols, width=0.55)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.15)
    axes[0, 0].set_ylim(0, 110)
    fig.suptitle("Perbandingan ResNet-18 vs ResNet-50 vs MobileNetV3-Small",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_perbandingan.png", dpi=150)
    plt.close(fig)


def plot_confusion(results, out_dir):
    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 5))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{r['name']}\nTest acc {r['test_acc'] * 100:.1f}%", fontweight="bold")
        ax.set_xticks(range(len(r["classes"])))
        ax.set_yticks(range(len(r["classes"])))
        ax.set_xticklabels(r["classes"], rotation=40, ha="right")
        ax.set_yticklabels(r["classes"])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Sebenarnya")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_confusion_matrix.png", dpi=150)
    plt.close(fig)


def print_table(results):
    print(f"\n{'Model':<20}{'Param(M)':>9}{'Ukuran(MB)':>11}{'Val acc':>9}{'Test acc':>9}"
          f"{'CPU ms':>8}{'Train(s)':>9}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<20}{r['params_m']:>9.1f}{r['size_mb']:>11.1f}{r['best_val_acc'] * 100:>8.1f}%"
              f"{r['test_acc'] * 100:>8.1f}%{r['cpu_latency_ms']:>8.1f}{r['train_time_s']:>9.0f}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="folder berisi train/ val/ test/")
    ap.add_argument("--out", default="hasil", help="folder output (model, chart, json)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3, help="epoch awal dengan backbone dibekukan")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2, help="set 0 jika error di Windows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-only", action="store_true", help="buat chart dari hasil.json yang sudah ada")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "hasil.json"

    if args.plot_only:
        results = json.loads(json_path.read_text())
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        loaders, classes = make_loaders(args.data, args.batch, args.img_size, args.workers)
        print("Kelas:", classes)
        print({k: len(v.dataset) for k, v in loaders.items()})

        results = []
        for name in MODEL_NAMES:
            results.append(train_one_model(name, loaders, classes, args, device, out_dir))
        json_path.write_text(json.dumps(results, indent=2))

    print_table(results)
    plot_curves(results, out_dir)
    plot_bars(results, out_dir)
    plot_confusion(results, out_dir)
    print(f"\nSelesai. Chart & model tersimpan di folder '{out_dir}/'")


if __name__ == "__main__":
    main()
