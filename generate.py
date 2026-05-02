import sys, os, math
import torch
import numpy as np

from src.cdvae_model import CDVAE, NUM_ELEMENTS, MAX_ATOMS, params_to_lattice

# ── Paths ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "cdvae_checkpoint_epoch_110.pt")

# ── Element data ─────────────────────────────────────
SYMBOL_TO_Z = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
    "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
    "Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,
    "Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
    "Pd":46,"Ag":47,"Ba":56,"La":57,"Hf":72,"W":74,"Pt":78,"Au":79,
    "Pb":82,"Bi":83,
}

Z2S = {v: k for k, v in SYMBOL_TO_Z.items()}

def to_json_safe(obj):
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]

    elif isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    else:
        return obj
# ── Main generate function ────────────────────────────
@torch.no_grad()
def generate(
    model,
    n_samples=1,
    device="cpu",
    composition=None,
    temperature=1.0,
    langevin_steps=30,
    step_size=1e-4
):
    model.eval()
    B = n_samples

    # ── Latent sampling ──
    z = torch.randn(B, model.latent, device=device) * temperature
    props = model.pred(z)

    # ── Composition control ──
    comp_probs = torch.zeros(B, NUM_ELEMENTS, device=device)
    if composition:
        for zn in composition:
            if 0 < zn < NUM_ELEMENTS:
                comp_probs[:, zn] = 1.0
        comp_probs = comp_probs / comp_probs.sum(-1, keepdim=True)
    else:
        comp_probs = props["comp"].softmax(-1)

    # ── Lattice ──
    lattice = params_to_lattice(props["lattice"])

    # ── Number of atoms ──
    n_ats = (props["n_atoms"].argmax(-1) + 1).clamp(min=2, max=MAX_ATOMS)

    N = n_ats.sum().item()
    a2b = torch.repeat_interleave(torch.arange(B, device=device), n_ats)

    # ── Initialize fractional coords ──
    fc = torch.rand(N, 3, device=device)

    # ── Element sampling ──
    if composition:
        allowed = torch.tensor(composition, device=device)
        types = allowed[torch.randint(0, len(allowed), (N,), device=device)]
    else:
        types = torch.multinomial(comp_probs[a2b], 1).squeeze(-1)

    # ── Langevin dynamics ──
    for j in range(model.n_sigs):
        sx = model.sigs_x[j]
        alpha = step_size * (sx**2) / (model.sigs_x[-1]**2)

        for _ in range(langevin_steps):
            sc, tl = model.score(
                fc, types, lattice, n_ats, z,
                sx.expand(B),
                model.sigs_a[j].expand(B)
            )

            fc = (fc + alpha * sc +
                  math.sqrt(2 * float(alpha)) * torch.randn_like(fc)) % 1.0

            types = tl.argmax(-1)

            # 🔒 enforce allowed elements again
            if composition:
                allowed = torch.tensor(composition, device=device)
                types = allowed[
                    torch.randint(0, len(allowed), (len(types),), device=device)
                ]

    # ── Build outputs ──
    out = []
    offset = 0

    for i in range(B):
        n = int(n_ats[i])

        fc_i = fc[offset:offset + n].cpu().numpy()
        t_i = types[offset:offset + n].cpu().tolist()

        # ── Remove invalid elements (safety) ──
        t_i = [t for t in t_i if t in Z2S]
        fc_i = fc_i[:len(t_i)]
        n = len(t_i)

        lat = lattice[i].cpu().numpy()
        cart = fc_i @ lat

        symbols = [Z2S[t] for t in t_i]

        # ── Lattice lengths ──
        a = np.linalg.norm(lat[0])
        b = np.linalg.norm(lat[1])
        c = np.linalg.norm(lat[2])

        out.append({
    "n_atoms": int(n),
    "atom_types": [int(x) for x in t_i],
    "symbols": symbols,

    # ✅ convert numpy → list
    "frac_coords": fc_i.tolist(),
    "cartesian": cart.tolist(),
    "lattice": lat.tolist(),

    "energy": float(props["energy"][i].item()),
    "ehull": float(max(0.0, props["ehull"][i].item())),
    "band_gap": float(props["bg"][i].item()),

    # ✅ convert tuple → list
    "abc": [round(a,3), round(b,3), round(c,3)],
})

        offset += int(n_ats[i])

    return to_json_safe(out)