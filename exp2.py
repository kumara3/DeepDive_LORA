from importlib.metadata import version
import os
from typing import Optional
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import tiktoken
from transformers import GPT2Model

# --- your utilities ---
from preprocess_data import (
    download_data,
    create_balanced_dataset,
    randomly_split_dataset,
    makespamDataset,
)
from Model_setup import GPT, calc_accuracy, train_classifier, plot_metrics
from Lora_setup import LoRAInjectedLinear

# =========================
# Config & utilities
# =========================
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

pkgs = ["matplotlib", "numpy", "tiktoken", "torch",
        "tensorflow", "pandas", "transformers"]
for p in pkgs:
    try:
        print(f"{p} version: {version(p)}")
    except Exception:
        pass

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = "sms_spam_collection.zip"
EXTRACTED = "sms_spam_collection"
DATA_TSV = Path(EXTRACTED) / "SMSSpamCollection.tsv"

MODEL_NAMES = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl",
}
CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "checkpoints"
BASE_WEIGHTS_PATH = "llmgpt_lora.pth"  # your stored base weights (pre-HF or your own)


# =========================
# Data prepration
# =========================
download_data(DATA_URL, ZIP_PATH, EXTRACTED, DATA_TSV)

df = pd.read_csv(DATA_TSV, sep="\t", header=None, names=["Label", "Text"])
df = create_balanced_dataset(df)
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

train_df, val_df, test_df = randomly_split_dataset(df, 0.7, 0.1)
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = makespamDataset("train.csv", max_length=None, tokenizer=tokenizer)
val_dataset   = makespamDataset("val.csv",   max_length=train_dataset.max_length, tokenizer=tokenizer)
test_dataset  = makespamDataset("test.csv",  max_length=train_dataset.max_length, tokenizer=tokenizer)

BATCH_SIZE = 8
NUM_WORKERS = 0

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, drop_last=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, drop_last=False
)

_ = GPT2Model.from_pretrained(MODEL_NAMES[CHOOSE_MODEL], cache_dir=CACHE_DIR).eval()


# =========================
# Model builders
# =========================
def build_base_config():
    cfg = BASE_CONFIG.copy()
    cfg.update(MODEL_CONFIGS[CHOOSE_MODEL])
    return cfg


def init_model_for_classification():
    """Create GPT, load base weights, replace head, move to device."""
    cfg = build_base_config()
    model = GPT(cfg)
    state = torch.load(BASE_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)

    # replace the output head for 2-class classification
    torch.manual_seed(SEED)
    model.out_head = torch.nn.Linear(in_features=cfg["emb_dim"], out_features=2)
    model.to(DEVICE)
    return model


def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad = False


def apply_lora_to_linear_layers(model: torch.nn.Module, rank: int, alpha: Optional[int] = None):
    """
    Recursively wrap every nn.Linear with LoRAInjectedLinear(rank, alpha).
    """
    if alpha is None:
        alpha = rank  # common default
    for name, mod in list(model.named_children()):
        if isinstance(mod, torch.nn.Linear):
            setattr(model, name, LoRAInjectedLinear(mod, rank=rank, alpha=alpha))
        else:
            apply_lora_to_linear_layers(mod, rank, alpha)


# =========================
# Experiment run
# =========================
def run_experiment(rank: int,
                   alpha: Optional[int],
                   trainable_layers: str,
                   lr: float = 5e-5,
                   weight_decay: float = 0.1,
                   num_epochs: int = 5,
                   eval_iter: int = 50,
                   eval_iter_val: int = 5):
    """
    Build a fresh model, set which layers are trainable, optionally inject LoRA,
    train, and return metrics.
    """
    start = time.time()

    model = init_model_for_classification()

    # set trainable parts
    if trainable_layers == "lora":
        freeze_all_params(model)
        apply_lora_to_linear_layers(model, rank=rank, alpha=alpha)
        for p in model.out_head.parameters():
            p.requires_grad = True
    elif trainable_layers == "last_block":
        freeze_all_params(model)
        for p in model.transformer_blocks[-1].parameters():
            p.requires_grad = True
        for p in model.out_head.parameters():
            p.requires_grad = True
    elif trainable_layers == "last_layer":
        freeze_all_params(model)
        for p in model.out_head.parameters():
            p.requires_grad = True
    elif trainable_layers == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Invalid trainable_layers='{trainable_layers}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # quick pre-train eval
    torch.manual_seed(SEED)
    pre_train_acc = calc_accuracy(train_loader, model, DEVICE, num_batches=8)
    pre_val_acc   = calc_accuracy(val_loader,   model, DEVICE, num_batches=8)
    pre_test_acc  = calc_accuracy(test_loader,  model, DEVICE, num_batches=8)

    # optimizer over trainable params only
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # train
    torch.manual_seed(SEED)
    tr_losses, va_losses, tr_accs, va_accs, examples_seen = train_classifier(
        model, train_loader, val_loader, optim, DEVICE,
        num_epochs=num_epochs, eval_iter=eval_iter, eval_iter_val=eval_iter_val
    )

    # final eval
    train_acc = calc_accuracy(train_loader, model, DEVICE)
    val_acc   = calc_accuracy(val_loader,   model, DEVICE)
    test_acc  = calc_accuracy(test_loader,  model, DEVICE)

    elapsed_min = (time.time() - start) / 60.0

    results = {
        "rank": int(rank),
        "alpha": int(alpha) if alpha is not None else int(rank),
        "trainable_layers": trainable_layers,
        "lr": lr,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "pre_train_acc": float(pre_train_acc),
        "pre_val_acc": float(pre_val_acc),
        "pre_test_acc": float(pre_test_acc),
        "final_train_acc": float(train_acc),
        "final_val_acc": float(val_acc),
        "final_test_acc": float(test_acc),
        "train_loss_last": float(tr_losses[-1]) if tr_losses else float("nan"),
        "val_loss_last": float(va_losses[-1]) if va_losses else float("nan"),
        "examples_seen": int(examples_seen),
        "minutes": elapsed_min,
    }

    
    try:
        epochs_tensor = torch.linspace(0, num_epochs, len(tr_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(tr_losses))
        plot_metrics(
            epochs_tensor,
            examples_seen_tensor,
            tr_losses,
            va_losses,
            label=f"loss_rank_{rank}_alpha_{alpha}_tl_{trainable_layers}",
        )
    except Exception as e:
        print(f"[WARN] Plotting failed for rank={rank}, alpha={alpha}, tl={trainable_layers}: {e}")

    return results, model


# =========================
# Params Sweeps
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-small (124M)",
        help="GPT model to use, choose from keys of MODEL_NAMES",
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA ranks to sweep over",
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha values to sweep over",
    )

    parser.add_argument(
        "--trainable_layers",
        nargs="+",
        type=str,
        default=["lora"],
        choices=["all", "last_block", "last_layer", "lora"],
        help="List of trainable layer modes to sweep over",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of training epochs per run",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training/validation/test",
    )

    parser.add_argument(
        "--eval_iter",
        type=int,
        default=50,
        help="Training iterations between train-eval logging",
    )

    parser.add_argument(
        "--eval_iter_val",
        type=int,
        default=5,
        help="Training iterations between val-eval logging",
    )

    args = parser.parse_args()


    run_rank = args.lora_rank
    run_alpha = args.lora_alpha
    default_tl = args.trainable_layers[0]
    epochs = args.num_epochs


    tl_rows = []
    tl_savedir = Path("trainable_layers_checkpoints")
    tl_savedir.mkdir(exist_ok=True, parents=True)

    for tl in args.trainable_layers:
        r = run_rank
        a = run_alpha
        print("\n" + "=" * 80)
        print(f"[TRAINABLE LAYERS SWEEP] rank={r}, alpha={a}, trainable_layers={tl}")
        print("=" * 80)

        res, model = run_experiment(
            rank=r,
            alpha=a,
            trainable_layers=tl,
            lr=5e-5,
            weight_decay=0.1,
            num_epochs=epochs,
            eval_iter=args.eval_iter,
            eval_iter_val=args.eval_iter_val,
        )
        tl_rows.append(res)

        ckpt_path = tl_savedir / f"llmgpt_rank{r}_alpha{a}_tl_{tl}.pth"
        try:
            torch.save(model.state_dict(), ckpt_path)
        except Exception as e:
            print(f"[WARN] Could not save checkpoint (trainable_layers sweep) for tl={tl}: {e}")

    df_tl = pd.DataFrame(tl_rows)

    for row in tl_rows:
        tl_name = row["trainable_layers"]
        per_tl_csv = f"tl_{tl_name}_results.csv"

        # if file doesn't exist, create with header
        if not os.path.exists(per_tl_csv):
            pd.DataFrame(columns=df_tl.columns).to_csv(per_tl_csv, index=False)
        df_row = pd.DataFrame([row])
        df_existing = pd.read_csv(per_tl_csv)
        df_new = pd.concat([df_existing, df_row], ignore_index=True)
        df_new.to_csv(per_tl_csv, index=False)
