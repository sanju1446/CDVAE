from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import traceback
import math

from generate import generate, SYMBOL_TO_Z
from src.cdvae_model import (
    CDVAE, NUM_ELEMENTS, MAX_ATOMS,
    params_to_lattice, build_periodic_graph, auto_cutoff
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "null",            # needed when index.html is opened as file://
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once ───────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CDVAE(
    hidden=256, latent=128,
    n_enc_layers=4, n_dec_layers=4,
    max_nb=12, n_sigmas=10,
    sigma_begin=0.5, sigma_end=0.005
).to(device)

ckpt = torch.load("cdvae_checkpoint_epoch_110.pt", map_location=device)
model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
model.eval()


# ── Request schemas ───────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    elements:    list[str]
    n_samples:   int
    temperature: float


class RecomputeRequest(BaseModel):
    # Cartesian coords of the CLUSTER only (outliers already removed by frontend)
    cartesian: list[list[float]]   # [[x,y,z], ...]  in Angstrom
    symbols:   list[str]           # ['Fe', 'Fe', 'O', ...]
    # Full unit cell lattice — unchanged after filtering
    lattice:   list[list[float]]   # [[a0,a1,a2],[b0,b1,b2],[c0,c1,c2]] row vectors


# ══════════════════════════════════════════════════════════════════════════════
#  Minimal PyG-compatible data container
#
#  PeriodicEncoder.forward() reads these exact attributes:
#    data.x          — atom types        (N, 1)  long
#    data.batch      — batch index       (N,)    long
#    data.ptr        — batch pointer     (B+1,)  long
#    data.edge_index — graph edges       (2, E)  long
#    data.edge_attr  — edge distances    (E,)    float
#    data.lattice    — lattice matrix    (B,3,3) float
#
#  Note: data.pos (frac coords) is read during training forward() for the
#  diffusion decoder but NOT used inside PeriodicEncoder itself — safe to
#  include anyway for completeness.
# ══════════════════════════════════════════════════════════════════════════════
class SimpleData:
    """Drop-in replacement for torch_geometric.data.Data — no PyG required."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_graph_data(frac: torch.Tensor,
                     atom_types: torch.Tensor,
                     lat: torch.Tensor,
                     dev: str) -> SimpleData:
    """
    Build the SimpleData graph object that PeriodicEncoder.forward() expects.

    Uses build_periodic_graph() from cdvae_model.py — the exact same function
    used at training time — so the graph structure is consistent.

    Args:
        frac        (N, 3)  fractional coordinates, values in [0, 1)
        atom_types  (N,)    atomic numbers, long tensor
        lat         (3, 3)  lattice row vectors in Angstrom
        dev                 torch device string

    Returns:
        SimpleData with x, pos, batch, ptr, edge_index, edge_attr, lattice
    """
    N = frac.shape[0]

    # Single-graph batch tensors (B = 1)
    batch     = torch.zeros(N, dtype=torch.long, device=dev)
    ptr       = torch.tensor([0, N], dtype=torch.long, device=dev)
    n_atoms_t = torch.tensor([N], dtype=torch.long, device=dev)
    lat_b     = lat.unsqueeze(0)                               # (1, 3, 3)

    # Build periodic graph — identical to training pipeline
    edge_index, edge_vecs, edge_dists = build_periodic_graph(
        frac,
        lat_b,
        n_atoms_t,
        cutoffs=auto_cutoff(lat_b, n_atoms_t).to(dev),
        max_neighbors=12
    )

    # edge_attr: PeriodicEncoder does data.edge_attr.float().view(-1)
    # so shape (E,) scalar distances is exactly right
    edge_attr = edge_dists.float()

    return SimpleData(
        x          = atom_types.view(-1, 1).long(),   # encoder: data.x.view(-1).long()
        pos        = frac,                             # frac coords (completeness)
        batch      = batch,
        ptr        = ptr,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        lattice    = lat_b,                            # (1, 3, 3)
    )


# ── /generate ─────────────────────────────────────────────────────────────────
@app.post("/generate")
def generate_api(req: GenerateRequest):
    try:
        composition = [
            SYMBOL_TO_Z[e.capitalize()]
            for e in req.elements
        ]
        results = generate(
            model,
            n_samples=req.n_samples,
            device=device,
            composition=composition,
            temperature=req.temperature
        )
        return results
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── /recompute ────────────────────────────────────────────────────────────────
@app.post("/recompute")
@torch.no_grad()
def recompute_api(req: RecomputeRequest):
    """
    Re-encode the bonded cluster through PeriodicEncoder → fresh z → pred(z).

    The frontend calls this endpoint ONLY when ALL three conditions are true:
        1. hadBonds  = true   — cluster has at least one detected bond
        2. intact    = true   — no element type was completely removed
        3. dropped   > 0      — at least one outlier atom was removed

    When hadBonds = false the frontend already marks the card INVALID and
    does NOT call /recompute — that case never reaches here.

    Flow:
        cluster cartesian coords (Angstrom)
                    |
                    v
        frac = cart @ lat_inv  (mod 1)          # back to fractional space
                    |
                    v
        build_graph_data()                       # same graph builder as training
                    |
                    v
        model.encoder(data)  ->  (mu, log_var)   # PeriodicEncoder
                    |
                    v  use mu (no noise at inference)
        model.pred(mu)  ->  energy, ehull, band_gap
    """
    try:
        n = len(req.symbols)

        if n < 2:
            return {"error": "Cluster must have at least 2 atoms."}

        # ── 1. Symbols → atomic number tensor ────────────────────────────────
        try:
            atom_types = torch.tensor(
                [SYMBOL_TO_Z[s.capitalize()] for s in req.symbols],
                dtype=torch.long,
                device=device
            )                                                  # (N,)
        except KeyError as e:
            return {"error": f"Unknown element symbol: {e}"}

        # ── 2. Lattice matrix (3, 3) ──────────────────────────────────────────
        lat = torch.tensor(
            req.lattice,
            dtype=torch.float32,
            device=device
        )                                                      # (3, 3)

        if lat.det().abs().item() < 1e-3:
            return {"error": "Degenerate lattice matrix (determinant ≈ 0)."}

        # ── 3. Cartesian → fractional coords ─────────────────────────────────
        cart    = torch.tensor(req.cartesian, dtype=torch.float32, device=device)
        lat_inv = torch.linalg.inv(lat)
        frac    = (cart @ lat_inv.T) % 1.0                    # (N, 3)

        # ── 4. Build graph data object ────────────────────────────────────────
        #   This is what was wrong in the previous version — PeriodicEncoder
        #   takes a data object, not raw tensors. We now build it correctly
        #   using the same build_periodic_graph() used at training time.
        data = build_graph_data(frac, atom_types, lat, device)

        # ── 5. Encode  →  (mu, log_var) ──────────────────────────────────────
        #   model.encoder  is  PeriodicEncoder
        #   It returns (mu, log_var), both shape (1, latent_dim)
        #   We use mu directly — no reparameterisation noise at inference
        mu, log_var = model.encoder(data)

        z = mu                                                 # (1, latent_dim)

        # ── 6. Predict properties from z ─────────────────────────────────────
        props = model.pred(z)

        return {
            "energy":   float(props["energy"][0].item()),
            "ehull":    float(max(0.0, props["ehull"][0].item())),
            "band_gap": float(props["bg"][0].item()),
        }

    except Exception as e:
        return {
            "error":     str(e),
            "traceback": traceback.format_exc()
        }