
"""
Train_Model.py — CDVAE Training v3
====================================
Changes from v2:
  ✓ W_LATTICE raised to 5.0 (was folded into W_AGG=1.0 → effectively 1.0)
    The lattice head now gets 5× the gradient pressure vs composition/n_atoms.
  ✓ KL warmup extended to 200 epochs (prevents posterior collapse early on)
  ✓ Gradient clipping tightened to 2.0 (was 5.0 — helps lattice stability)
  ✓ Diagnostic: log predicted a,b,c norms each epoch to track convergence
  ✓ All other logic unchanged
"""
import os, sys, json, shutil, subprocess
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_BASE     = "/kaggle/input/datasets/builderbob05/cdvae3-1"
DATA_PATH      = f"{INPUT_BASE}/dataset_files"
SRC_PATH       = "/kaggle/working/datasets/builderbob05/cdvae3-1/src"
LOCAL_CKPT_DIR = "/kaggle/working/checkpoints"
PUSH_DIR       = "/kaggle/working/ckpt_push"
PULL_DIR       = "/kaggle/working/ckpt_pull"
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)

sys.path.insert(0, SRC_PATH)
from train_dataset import CrystalGraphDataset
from cdvae_model   import CDVAE

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
EPOCHS       = 300
LR           = 1e-4
WEIGHT_DECAY = 1e-5
LR_MIN       = 1e-6
KL_MAX       = 0.01
KL_WARMUP    = 200     # FIX: was 100 — extended to reduce posterior collapse
SAVE_EVERY   = 5
MAX_ATOMS    = 50

# Loss weights
W_KL     = 0.001
W_COMP   = 1.0         # composition head
W_LATTICE= 5.0         # FIX: was effectively 1.0 (folded in W_AGG) → now 5.0
W_NATOMS = 1.0         # n_atoms head
W_DEC    = 2.0
W_ENERGY = 0.1
W_EHULL  = 0.05
W_BG     = 0.02

# Model architecture
HIDDEN_DIM   = 256
LATENT_DIM   = 128
N_ENC_LAYERS = 4
N_DEC_LAYERS = 4
MAX_NEIGHBORS= 12
N_SIGMAS     = 10
SIGMA_BEGIN  = 0.5
SIGMA_END    = 0.005

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "builderbob05")
KAGGLE_DATASET  = f"{KAGGLE_USERNAME}/epochs05"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device : {DEVICE}")
print(f"Data   : {DATA_PATH}")

# ── Kaggle helpers ─────────────────────────────────────────────────────────────
def _kaggle_ok():
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

def _write_meta():
    meta = {"title":"epochs05","id":KAGGLE_DATASET,
            "licenses":[{"name":"CC0-1.0"}]}
    with open(os.path.join(PUSH_DIR,"dataset-metadata.json"),"w") as f:
        json.dump(meta, f)

def push_kaggle(path):
    if not _kaggle_ok():
        print("Kaggle creds missing — saved locally only"); return
    try:
        os.makedirs(PUSH_DIR, exist_ok=True)
        for f in os.listdir(PUSH_DIR):
            if f.endswith(".pt"): os.remove(os.path.join(PUSH_DIR,f))
        shutil.copy2(path, PUSH_DIR); _write_meta()
        r = subprocess.run(
            ["kaggle","datasets","version","-p",PUSH_DIR,
             "-m",os.path.basename(path),"--dir-mode","zip"],
            capture_output=True, text=True)
        print(f"Kaggle push: {'OK' if r.returncode==0 else r.stderr.strip()}")
    except Exception as e:
        print(f"Kaggle push error: {e}")

def _epoch_num(p):
    try: return int(os.path.basename(p).split("epoch_")[1].replace(".pt",""))
    except: return -1

def latest_local():
    ckpts = [os.path.join(LOCAL_CKPT_DIR,f)
             for f in os.listdir(LOCAL_CKPT_DIR)
             if f.startswith("cdvae_checkpoint_epoch") and f.endswith(".pt")]
    if not ckpts: return None
    ckpts.sort(key=_epoch_num)
    print(f"Local checkpoint: {ckpts[-1]}"); return ckpts[-1]

def latest_kaggle():
    if not _kaggle_ok(): return None
    try:
        if os.path.exists(PULL_DIR): shutil.rmtree(PULL_DIR)
        os.makedirs(PULL_DIR, exist_ok=True)
        r = subprocess.run(
            ["kaggle","datasets","download",KAGGLE_DATASET,
             "-p",PULL_DIR,"--unzip"],
            capture_output=True, text=True)
        if r.returncode != 0: return None
        ckpts = []
        for root,_,files in os.walk(PULL_DIR):
            for f in files:
                if f.startswith("cdvae_checkpoint_epoch") and f.endswith(".pt"):
                    ckpts.append(os.path.join(root,f))
        if not ckpts: return None
        ckpts.sort(key=_epoch_num)
        print(f"Kaggle checkpoint: {ckpts[-1]}"); return ckpts[-1]
    except Exception as e:
        print(f"Kaggle pull error: {e}"); return None

def save_checkpoint(state, epoch):
    fname = f"cdvae_checkpoint_epoch_{epoch}.pt"
    path  = os.path.join(LOCAL_CKPT_DIR, fname)
    torch.save(state, path)
    print(f"Saved: {path}")
    push_kaggle(path)

# ── Data ───────────────────────────────────────────────────────────────────────
dataset       = CrystalGraphDataset(DATA_PATH)
valid_indices = list(range(len(dataset)))
dataset       = Subset(dataset, valid_indices)

print(f"Dataset      : {len(dataset)} samples")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Steps/epoch  : {len(dataset) // BATCH_SIZE}")

sample = dataset[0]
print(f"Energy labels:")
print(f"  formation  : {'OK' if not torch.isnan(sample.y).all() else 'MISSING'}")
print(f"  e_above_hull: {'OK' if not torch.isnan(sample.e_above_hull).all() else 'MISSING'}")
print(f"  band_gap   : {'OK' if not torch.isnan(sample.band_gap).all() else 'MISSING'}")
if torch.isnan(sample.y).all():
    raise SystemExit("Rebuild dataset with build_dataset.py first.")

torch.backends.cudnn.benchmark = True
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True,
    follow_batch=["x"],
)

# ── Model ──────────────────────────────────────────────────────────────────────
print("Initialising CDVAE (no torch_scatter, auto cutoff)...")
model = CDVAE(
    hidden       = HIDDEN_DIM,
    latent       = LATENT_DIM,
    n_enc_layers = N_ENC_LAYERS,
    n_dec_layers = N_DEC_LAYERS,
    max_nb       = MAX_NEIGHBORS,
    n_sigmas     = N_SIGMAS,
    sigma_begin  = SIGMA_BEGIN,
    sigma_end    = SIGMA_END,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters : {n_params:,}")

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
history   = []
start_epoch = 0

# ── Resume ─────────────────────────────────────────────────────────────────────
ckpt_path = latest_local() or latest_kaggle()
if ckpt_path:
    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    miss, _ = model.load_state_dict(ckpt["model_state"], strict=False)
    if miss: print(f"Reinitialized {len(miss)} layers")
    try: optimizer.load_state_dict(ckpt["optimizer_state"])
    except Exception as e: print(f"Optimizer not restored: {e}")
    try: scheduler.load_state_dict(ckpt["scheduler_state"])
    except: pass
    start_epoch = ckpt["epoch"]
    history     = ckpt.get("history", [])
    print(f"Resumed from epoch {start_epoch}")
else:
    print("Starting fresh.")

if start_epoch >= EPOCHS:
    raise SystemExit(0)

print(f"\nTraining epoch {start_epoch + 1} → {EPOCHS}\n")
print(f"Loss weights: KL={W_KL}, comp={W_COMP}, lattice={W_LATTICE}, "
      f"n_atoms={W_NATOMS}, decoder={W_DEC}, energy={W_ENERGY}")

# ── KL annealing ───────────────────────────────────────────────────────────────
def kl_weight(epoch):
    return min(KL_MAX, KL_MAX * (epoch + 1) / KL_WARMUP)

# ── Training loop ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    totals = {k: 0.0 for k in [
        "loss","kl","comp","lattice","n_atoms",
        "score","type","decoder","energy","ehull","bg"]}
    kl_w = kl_weight(epoch)

    # Track predicted lattice param norms for diagnostics
    lat_a_preds = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        batch  = batch.to(DEVICE)
        losses = model(batch)

        # FIX: lattice now has its own weight W_LATTICE instead of being
        # lumped into W_AGG with comp and n_atoms.
        total = (
            kl_w       * losses["kl"]      +
            W_COMP     * losses["comp"]     +
            W_LATTICE  * losses["lattice"]  +
            W_NATOMS   * losses["n_atoms"]  +
            W_DEC      * losses["decoder"]  +
            W_ENERGY   * losses["energy"]   +
            W_EHULL    * losses["ehull"]    +
            W_BG       * losses["bg"]
        )

        optimizer.zero_grad()
        total.backward()
        # FIX: tighter grad clip — lattice head is sensitive
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        for k in totals:
            if k == "loss": totals[k] += total.item()
            elif k in losses: totals[k] += losses[k].item()

        # Diagnostic: sample predicted a for this batch
        with torch.no_grad():
            z_diag  = torch.randn(1, LATENT_DIM, device=DEVICE)
            p_diag  = model.pred(z_diag)["lattice"]
            import torch.nn.functional as F_
            a_diag  = (F_.softplus(p_diag[0, 0]) + 0.5).item()
            lat_a_preds.append(a_diag)

        pbar.set_postfix(
            loss    = f"{total.item():.4f}",
            lat_l   = f"{losses['lattice'].item():.4f}",
            score   = f"{losses['score'].item():.4f}",
            energy  = f"{losses['energy'].item():.4f}",
        )

    n    = len(loader)
    avgs = {k: v / n for k, v in totals.items()}

    mean_a_pred = sum(lat_a_preds) / len(lat_a_preds)
    history.append({"epoch": epoch + 1, **avgs, "mean_a_pred": mean_a_pred})

    print(
        f"Epoch {epoch+1:3d}/{EPOCHS} | loss {avgs['loss']:.4f} | "
        f"kl {avgs['kl']:.4f} | comp {avgs['comp']:.4f} | "
        f"lat {avgs['lattice']:.4f} | score {avgs['score']:.4f} | "
        f"type {avgs['type']:.4f} | energy {avgs['energy']:.4f} | "
        f"mean_a_pred {mean_a_pred:.2f} Å | "   # ← watch this converge to ~6 Å
        f"lr {optimizer.param_groups[0]['lr']:.2e}"
    )

    scheduler.step()

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        save_checkpoint({
            "epoch"          : epoch + 1,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss"           : avgs["loss"],
            "history"        : history,
        }, epoch + 1)

# ── Final save ─────────────────────────────────────────────────────────────────
final = os.path.join(LOCAL_CKPT_DIR, "cdvae_model_final.pt")
torch.save(model.state_dict(), final)
push_kaggle(final)

with open(os.path.join(LOCAL_CKPT_DIR,"training_history.json"),"w") as f:
    json.dump(history, f, indent=2)

print("\nTraining complete.")