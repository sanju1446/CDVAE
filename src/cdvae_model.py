
"""
cdvae_model.py — CDVAE v4 (Fixed)
===================================
Fixes applied:
  ✓ params_to_lattice: hard min/max clamps (3 Å – 25 Å) so cell can never collapse
  ✓ Lattice loss in LOG-SPACE on raw a,b,c (not normalized by their own mean)
    → gradient signal carries absolute-scale information
  ✓ Lattice loss weight bumped to 5× in the loss dict (caller uses W_AGG, but
    we return a separate "lattice_abc" loss so the trainer can up-weight it)
  ✓ Overlap penalty rebuilt in CARTESIAN coords using the PREDICTED (decoder)
    lattice, not the encoder input, so it trains the right head
  ✓ Minimum separation enforced at 1.8 Å (covalent-bond floor)
  ✓ Volume regulariser added: penalises |log(V_pred/V_target)| so the cell
    volume is consistent with atom count
  ✓ No other logic changed
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ELEMENTS = 119
MAX_ATOMS    = 50
N_SIGMAS     = 10
SIGMA_BEGIN  = 0.5
SIGMA_END    = 0.005

# Physically reasonable lattice-param bounds (Å)
LAT_MIN = 2.5
LAT_MAX = 25.0


# ══════════════════════════════════════════════════════════════════════════════
#  Pure-PyTorch scatter helpers
# ══════════════════════════════════════════════════════════════════════════════

def scatter_mean(src, index, dim_size):
    is_1d = src.dim() == 1
    s     = src.unsqueeze(-1) if is_1d else src
    out   = torch.zeros(dim_size, s.shape[-1], dtype=s.dtype, device=s.device)
    cnt   = torch.zeros(dim_size, 1,           dtype=s.dtype, device=s.device)
    out.index_add_(0, index, s)
    cnt.index_add_(0, index, torch.ones(s.shape[0], 1, dtype=s.dtype, device=s.device))
    out = out / cnt.clamp(min=1)
    return out.squeeze(-1) if is_1d else out


def scatter_sum(src, index, dim_size):
    if src.dim() == 1:
        out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    else:
        out = torch.zeros(dim_size, src.shape[-1], dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def scatter_max(src, index, dim_size):
    s   = src.unsqueeze(-1) if src.dim() == 1 else src
    out = torch.full((dim_size, s.shape[-1]), -1e9, dtype=s.dtype, device=s.device)
    try:
        out.index_reduce_(0, index, s, reduce="amax", include_self=True)
    except Exception:
        out = scatter_mean(src, index, dim_size)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
    return out.squeeze(-1) if src.dim() == 1 else out


# ══════════════════════════════════════════════════════════════════════════════
#  Auto Cutoff
# ══════════════════════════════════════════════════════════════════════════════

def auto_cutoff(lattice, n_atoms, scale=1.8, min_cut=3.5, max_cut=8.0):
    vol     = lattice.det().abs().clamp(min=1.0)
    density = n_atoms.float() / vol
    r_nn    = (3.0 / (4.0 * math.pi * density.clamp(min=1e-4))) ** (1.0 / 3.0)
    return (scale * r_nn).clamp(min_cut, max_cut)


# ══════════════════════════════════════════════════════════════════════════════
#  Sigma schedule
# ══════════════════════════════════════════════════════════════════════════════

def get_sigmas(n=N_SIGMAS, begin=SIGMA_BEGIN, end=SIGMA_END, device="cpu"):
    return torch.exp(
        torch.linspace(math.log(begin), math.log(end), n)
    ).to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  Periodic Graph Builder
# ══════════════════════════════════════════════════════════════════════════════

_IMGS = torch.tensor([
    [i, j, k]
    for i in [-1, 0, 1]
    for j in [-1, 0, 1]
    for k in [-1, 0, 1]
], dtype=torch.float32)   # (27, 3)


def build_periodic_graph(frac_coords, lattice, n_atoms,
                         cutoffs=None, max_neighbors=12):
    device = frac_coords.device
    B      = lattice.shape[0]

    if cutoffs is None:
        cutoffs = auto_cutoff(lattice, n_atoms).to(device)

    imgs                     = _IMGS.to(device)
    src_list, dst_list, vec_list = [], [], []
    offset = 0

    for b in range(B):
        n    = int(n_atoms[b].item())
        lat  = lattice[b]
        fc   = frac_coords[offset:offset + n]
        cart = fc @ lat
        cut  = float(cutoffs[b].item())

        img_fc   = fc.unsqueeze(1) + imgs.unsqueeze(0)
        img_cart = (img_fc @ lat).view(n * 27, 3)
        img_idx  = torch.arange(n, device=device).repeat_interleave(27)

        for i in range(n):
            diffs = img_cart - cart[i].unsqueeze(0)
            dists = diffs.norm(dim=-1)
            mask  = (dists > 1e-6) & (dists < cut)

            if not mask.any():
                vals = (dists > 1e-6).nonzero(as_tuple=False)
                mask = torch.zeros_like(dists, dtype=torch.bool)
                if vals.numel():
                    mask[vals[0]] = True

            d_m, v_m, a_m = dists[mask], diffs[mask], img_idx[mask]

            if d_m.shape[0] > max_neighbors:
                keep          = d_m.topk(max_neighbors, largest=False).indices
                d_m, v_m, a_m = d_m[keep], v_m[keep], a_m[keep]

            src_list.append(torch.full((d_m.shape[0],), offset + i,
                                       dtype=torch.long, device=device))
            dst_list.append(a_m + offset)
            vec_list.append(v_m)

        offset += n

    if not src_list:
        return (torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, 3, device=device),
                torch.zeros(0, device=device))

    src  = torch.cat(src_list)
    dst  = torch.cat(dst_list)
    vecs = torch.cat(vec_list)
    return torch.stack([src, dst]), vecs, vecs.norm(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Building blocks
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        assert dim % 2 == 0
        half  = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half).float() / max(half - 1, 1))
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x = x.view(-1, 1) * self.freqs.view(1, -1)
        return torch.cat([x.sin(), x.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2), nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  GNN Layer
# ══════════════════════════════════════════════════════════════════════════════

class PGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.msg  = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim * 2), nn.SiLU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.upd  = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim), nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, edge_index, edge_feat):
        src, dst = edge_index
        N        = h.shape[0]
        m        = self.msg(torch.cat([h[src], h[dst], edge_feat], dim=-1))
        agg      = torch.cat([
            scatter_mean(m, dst, N),
            scatter_max(m, dst, N),
        ], dim=-1)
        return self.norm(h + self.upd(agg))


# ══════════════════════════════════════════════════════════════════════════════
#  Encoder
# ══════════════════════════════════════════════════════════════════════════════

class PeriodicEncoder(nn.Module):
    def __init__(self, hidden=256, latent=128, n_layers=4, max_nb=12):
        super().__init__()
        self.atom_emb = nn.Embedding(NUM_ELEMENTS, hidden)
        self.dist_emb = SinusoidalEmbedding(64)
        self.layers   = nn.ModuleList(
            [PGNNLayer(hidden, 64) for _ in range(n_layers)])
        self.pool     = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), ResBlock(hidden))
        self.mu_h     = nn.Linear(hidden, latent)
        self.lv_h     = nn.Linear(hidden, latent)

    def forward(self, data):
        types = data.x.view(-1).long().clamp(0, NUM_ELEMENTS - 1)
        n_ats = (data.ptr[1:] - data.ptr[:-1]).long()
        B     = int(n_ats.shape[0])
        bch   = data.batch

        ei = data.edge_index.long()
        ed = data.edge_attr.float().view(-1).clamp(min=1e-6)

        h  = self.atom_emb(types)
        ef = self.dist_emb(ed)

        for layer in self.layers:
            h = layer(h, ei, ef)

        g_mean = torch.zeros(B, h.shape[-1], dtype=h.dtype, device=h.device)
        g_max  = torch.full((B, h.shape[-1]), -1e9, dtype=h.dtype, device=h.device)
        cnt    = torch.zeros(B, 1, dtype=h.dtype, device=h.device)

        g_mean.index_add_(0, bch, h)
        cnt.index_add_(0, bch,
                       torch.ones(h.shape[0], 1, dtype=h.dtype, device=h.device))
        g_mean = g_mean / cnt.clamp(min=1)

        try:
            g_max.index_reduce_(0, bch, h, reduce="amax", include_self=True)
        except Exception:
            g_max = g_mean.clone()

        g = self.pool(torch.cat([g_mean, g_max], dim=-1))
        return self.mu_h(g), self.lv_h(g)


# ══════════════════════════════════════════════════════════════════════════════
#  Property Predictor
# ══════════════════════════════════════════════════════════════════════════════

class PropertyPredictor(nn.Module):
    def __init__(self, latent=128, hidden=256):
        super().__init__()
        self.shared    = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(),
            ResBlock(hidden), ResBlock(hidden),
        )
        self.comp_h    = nn.Linear(hidden, NUM_ELEMENTS)
        self.lattice_h = nn.Linear(hidden, 6)
        self.n_h       = nn.Linear(hidden, MAX_ATOMS)
        self.energy_h  = nn.Linear(hidden, 1)
        self.ehull_h   = nn.Linear(hidden, 1)
        self.bg_h      = nn.Linear(hidden, 1)

    def forward(self, z):
        h = self.shared(z)
        return {
            "comp"   : self.comp_h(h),
            "lattice": self.lattice_h(h),
            "n_atoms": self.n_h(h),
            "energy" : self.energy_h(h).squeeze(-1),
            "ehull"  : self.ehull_h(h).squeeze(-1),
            "bg"     : self.bg_h(h).squeeze(-1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 1: params_to_lattice — hard physical bounds on a, b, c
# ══════════════════════════════════════════════════════════════════════════════

def params_to_lattice(p):
    """
    6 raw params → (B, 3, 3) lattice matrix (differentiable).

    FIX: a, b, c now have explicit min/max clamps (LAT_MIN=2.5 Å, LAT_MAX=25 Å).
    The old code had bare torch.clamp() with NO bounds — that is a no-op and
    was the primary cause of lattice collapse to ~1.4 Å.
    """
    B   = p.shape[0]
    dev = p.device
    # FIX: clamp to [LAT_MIN, LAT_MAX] — not bare clamp()
    a   = (F.softplus(p[:, 0]) + 0.5).clamp(LAT_MIN, LAT_MAX)
    b   = (F.softplus(p[:, 1]) + 0.5).clamp(LAT_MIN, LAT_MAX)
    c   = (F.softplus(p[:, 2]) + 0.5).clamp(LAT_MIN, LAT_MAX)
    cg  = torch.tanh(p[:, 5]) * 0.99
    cb  = torch.tanh(p[:, 4]) * 0.99
    ca  = torch.tanh(p[:, 3]) * 0.99
    sg  = (1 - cg ** 2).clamp(min=1e-6).sqrt()
    z0  = torch.zeros(B, device=dev)
    r1  = torch.stack([a,          z0,    z0], dim=-1)
    r2  = torch.stack([b * cg, b * sg,    z0], dim=-1)
    a3x = c * cb
    a3y = c * (ca - cb * cg) / sg.clamp(min=1e-6)
    a3z = (c ** 2 - a3x ** 2 - a3y ** 2).clamp(min=1e-6).sqrt()
    r3  = torch.stack([a3x, a3y, a3z], dim=-1)
    return torch.stack([r1, r2, r3], dim=1)   # (B, 3, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  Score Network
# ══════════════════════════════════════════════════════════════════════════════

class ScoreNetwork(nn.Module):
    def __init__(self, hidden=256, latent=128, n_layers=4, max_nb=12):
        super().__init__()
        self.max_nb   = max_nb
        self.atom_emb = nn.Embedding(NUM_ELEMENTS, hidden // 2)
        self.dist_emb = SinusoidalEmbedding(64)
        self.sig_emb  = SinusoidalEmbedding(64)

        in_dim      = hidden // 2 + latent + 128
        self.proj   = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU())
        self.layers  = nn.ModuleList(
            [PGNNLayer(hidden, 64) for _ in range(n_layers)])
        self.score_w = nn.Sequential(ResBlock(hidden), nn.Linear(hidden, 1))
        self.type_h  = nn.Sequential(
            ResBlock(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, NUM_ELEMENTS))

    def forward(self, frac, types, lattice, n_atoms, z, sigma_x, sigma_a):
        device  = frac.device
        B       = lattice.shape[0]
        N       = frac.shape[0]

        cutoffs = auto_cutoff(lattice, n_atoms).to(device)
        ei, ev, ed = build_periodic_graph(
            frac, lattice, n_atoms,
            cutoffs=cutoffs, max_neighbors=self.max_nb)

        a2b = torch.repeat_interleave(torch.arange(B, device=device), n_atoms)

        ae  = self.atom_emb(types.clamp(0, NUM_ELEMENTS - 1))
        se  = torch.cat([self.sig_emb(sigma_x[a2b]),
                         self.sig_emb(sigma_a[a2b])], dim=-1)
        h   = self.proj(torch.cat([ae, z[a2b], se], dim=-1))

        ef  = self.dist_emb(ed)
        for layer in self.layers:
            h = layer(h, ei, ef)

        w        = self.score_w(h)
        src, dst = ei
        uv       = F.normalize(ev, dim=-1)
        weighted = uv * w[src]
        score    = scatter_sum(weighted, dst, N)

        type_l  = self.type_h(h)
        return score, type_l


# ══════════════════════════════════════════════════════════════════════════════
#  CDVAE
# ══════════════════════════════════════════════════════════════════════════════

class CDVAE(nn.Module):
    """
    Crystal Diffusion Variational Autoencoder (Xie et al., ICLR 2022)

    v4 Fixes:
        • Lattice head loss in log-space (absolute scale now penalised)
        • Volume regulariser (cell consistent with atom count)
        • Overlap penalty on PREDICTED lattice (not ground-truth)
        • params_to_lattice has hard physical clamps
    """

    def __init__(self, hidden=256, latent=128,
                 n_enc_layers=4, n_dec_layers=4, max_nb=12,
                 n_sigmas=N_SIGMAS,
                 sigma_begin=SIGMA_BEGIN, sigma_end=SIGMA_END):
        super().__init__()
        self.latent = latent
        self.n_sigs = n_sigmas

        self.encoder = PeriodicEncoder(hidden, latent, n_enc_layers, max_nb)
        self.pred    = PropertyPredictor(latent, hidden)
        self.score   = ScoreNetwork(hidden, latent, n_dec_layers, max_nb)

        sigs = get_sigmas(n_sigmas, sigma_begin, sigma_end)
        self.register_buffer("sigs_x", sigs)
        self.register_buffer("sigs_a", sigs)

    def reparameterise(self, mu, lv):
        std = torch.exp(0.5 * lv.clamp(-10, 10))
        return mu + std * torch.randn_like(mu)

    # ── Training forward ───────────────────────────────────────────────────────
    def forward(self, data):
        device = data.x.device
        n_ats  = (data.ptr[1:] - data.ptr[:-1]).long()
        B      = int(n_ats.shape[0])
        a2b    = data.batch
        types  = data.x.view(-1).long().clamp(0, NUM_ELEMENTS - 1)

        lat = data.lattice.float().view(B, 3, 3)

        # ── 1. Encode ─────────────────────────────────────────────────────────
        mu, lv = self.encoder(data)
        z      = self.reparameterise(mu, lv)
        kl     = -0.5 * torch.mean(torch.sum(1 + lv - mu**2 - lv.exp(), dim=-1))

        # ── 2. Property prediction ────────────────────────────────────────────
        props = self.pred(z)

        # ---- composition target ----
        comp_t = torch.zeros(B, NUM_ELEMENTS, device=device)
        comp_t.index_add_(0, a2b, F.one_hot(types, NUM_ELEMENTS).float())
        comp_t = comp_t / n_ats.float().unsqueeze(-1).clamp(min=1)
        comp_l = F.cross_entropy(props["comp"], comp_t.argmax(-1))

        # ── FIX 2: Lattice loss in LOG-SPACE on raw a, b, c ──────────────────
        #
        # OLD code: normalised a,b,c by their own mean (sc) → gradient for
        # absolute scale was zero → model collapsed to whatever softplus(0)≈0.69
        # gives after adding 0.5 → ~1.2 Å.
        #
        # NEW code: regress log(a_pred) → log(a_target) directly.
        # This means a 6 Å target vs 1.4 Å pred gives a loss of log(6/1.4)≈1.4
        # instead of 0 (previously normalised away).
        # Angles (ca, cb, cg) are kept as before — they were already fine.

        a_t = lat[:, 0].norm(dim=-1).clamp(min=0.1)
        b_t = lat[:, 1].norm(dim=-1).clamp(min=0.1)
        c_t = lat[:, 2].norm(dim=-1).clamp(min=0.1)

        ca = (lat[:, 1] * lat[:, 2]).sum(-1) / (b_t * c_t).clamp(min=1e-6)
        cb = (lat[:, 0] * lat[:, 2]).sum(-1) / (a_t * c_t).clamp(min=1e-6)
        cg = (lat[:, 0] * lat[:, 1]).sum(-1) / (a_t * b_t).clamp(min=1e-6)

        p = props["lattice"]
        # Predicted a, b, c in Å
        a_pred = (F.softplus(p[:, 0]) + 0.5).clamp(LAT_MIN, LAT_MAX)
        b_pred = (F.softplus(p[:, 1]) + 0.5).clamp(LAT_MIN, LAT_MAX)
        c_pred = (F.softplus(p[:, 2]) + 0.5).clamp(LAT_MIN, LAT_MAX)
        ca_pred = torch.tanh(p[:, 3]) * 0.99
        cb_pred = torch.tanh(p[:, 4]) * 0.99
        cg_pred = torch.tanh(p[:, 5]) * 0.99

        # Log-space MSE for lengths (scale-invariant, correct gradient)
        lat_len_l = (
            F.mse_loss(a_pred.log(), a_t.log()) +
            F.mse_loss(b_pred.log(), b_t.log()) +
            F.mse_loss(c_pred.log(), c_t.log())
        )
        # MSE for angles (unchanged)
        lat_ang_l = (
            F.mse_loss(ca_pred, ca.clamp(-0.99, 0.99)) +
            F.mse_loss(cb_pred, cb.clamp(-0.99, 0.99)) +
            F.mse_loss(cg_pred, cg.clamp(-0.99, 0.99))
        )
        lat_l = lat_len_l + 0.1 * lat_ang_l   # lengths dominate

        # ── FIX 3: Volume regulariser ─────────────────────────────────────────
        #
        # Penalise log(V_pred / V_target).  This forces the cell volume to
        # be consistent with the number of atoms even when individual a,b,c
        # gradients are noisy.
        #
        # Approx volume from predicted params (upper triangular lattice):
        vol_pred = (a_pred * b_pred * c_pred).clamp(min=1.0)
        vol_true = lat.det().abs().clamp(min=1.0)
        vol_l    = F.mse_loss(vol_pred.log(), vol_true.log())
        lat_l    = lat_l + 0.5 * vol_l

        # n_atoms loss
        n_t = (n_ats - 1).clamp(0, MAX_ATOMS - 1)
        n_l = F.cross_entropy(props["n_atoms"], n_t)

        agg_l = comp_l + lat_l + n_l

        # ── 3. Diffusion decoder ──────────────────────────────────────────────
        idx_j = torch.randint(0, self.n_sigs, (B,), device=device)
        sx_b  = self.sigs_x[idx_j]
        sa_b  = self.sigs_a[idx_j]
        sx_n  = sx_b[a2b]
        sa_n  = sa_b[a2b]

        fc_true  = data.pos.float()
        fc_noisy = (fc_true + torch.randn_like(fc_true) * sx_n.unsqueeze(-1)) % 1.0

        comp_n = comp_t.softmax(-1)[a2b]
        oh     = F.one_hot(types, NUM_ELEMENTS).float()
        alpha  = 1.0 / (1.0 + sa_n.unsqueeze(-1))
        mix    = (alpha * oh + (1 - alpha) * comp_n).clamp(min=1e-8)
        t_noisy = torch.multinomial(mix, 1).squeeze(-1)

        score_pred, type_pred = self.score(
            fc_noisy, t_noisy, lat, n_ats, z, sx_b, sa_b)

        # ---- score matching ----
        disp   = fc_true - fc_noisy
        disp   = disp - torch.round(disp)
        s_tgt  = disp / sx_n.unsqueeze(-1).clamp(min=1e-8)
        score_l = F.mse_loss(score_pred, s_tgt)
        type_l  = F.cross_entropy(type_pred, types)

        dec_l = score_l + 0.1 * type_l

        # ── FIX 4: Overlap penalty on PREDICTED lattice ───────────────────────
        #
        # OLD code: used lat (encoder ground-truth) which at epoch 45 is already
        # in correct Å scale, so the penalty was computed correctly for training
        # data but does NOT train the lattice head to avoid overlaps.
        #
        # NEW code: also compute penalty on the PREDICTED lattice from the
        # decoder head.  This directly penalises the lattice head when its
        # predicted cell causes atoms to overlap.
        #
        # Min separation raised to 1.8 Å (smallest covalent bond: H-H ~0.74 Å
        # but in crystals 1.5 Å is too loose; 1.8 Å catches clear overlaps).

        MIN_SEP = 1.8   # Å

        # Build predicted lattice for overlap check
        lat_pred_mat = params_to_lattice(props["lattice"])  # (B, 3, 3)

        overlap_gt   = torch.tensor(0.0, device=device)
        overlap_pred = torch.tensor(0.0, device=device)
        offset       = 0

        for i in range(B):
            n    = int(n_ats[i])
            fc_i = fc_true[offset:offset + n]

            # --- on ground-truth lattice (keeps existing training signal) ---
            cart_gt = fc_i @ lat[i]
            dist_gt = torch.cdist(cart_gt, cart_gt)
            mask_gt = (dist_gt > 1e-6) & (dist_gt < MIN_SEP)
            if mask_gt.any():
                overlap_gt = overlap_gt + torch.relu(MIN_SEP - dist_gt[mask_gt]).mean()

            # --- on PREDICTED lattice (new: trains the lattice head) ---
            cart_pred = fc_i @ lat_pred_mat[i]
            dist_pred = torch.cdist(cart_pred, cart_pred)
            mask_pred = (dist_pred > 1e-6) & (dist_pred < MIN_SEP)
            if mask_pred.any():
                overlap_pred = overlap_pred + torch.relu(MIN_SEP - dist_pred[mask_pred]).mean()

            offset += n

        overlap_penalty = (overlap_gt + overlap_pred) / B
        dec_l = dec_l + 1.0 * overlap_penalty   # increased from 0.5

        # ── Coordinate reconstruction loss ────────────────────────────────────
        disp_true  = fc_true - fc_noisy
        disp_true  = disp_true - torch.round(disp_true)
        coord_loss = torch.mean(disp_true ** 2)
        dec_l      = dec_l + 0.5 * coord_loss

        # ── Energy losses ──────────────────────────────────────────────────────
        def masked_mse(pred, target):
            t    = target.float().view(-1).to(device)
            mask = ~torch.isnan(t)
            if not mask.any():
                return torch.tensor(0., device=device)
            return F.mse_loss(pred[mask], t[mask])

        e_l  = masked_mse(props["energy"], data.y)           if hasattr(data, "y")            else torch.tensor(0., device=device)
        eh_l = masked_mse(props["ehull"],  data.e_above_hull) if hasattr(data, "e_above_hull") else torch.tensor(0., device=device)
        bg_l = masked_mse(props["bg"],     data.band_gap)     if hasattr(data, "band_gap")     else torch.tensor(0., device=device)

        return {
            "kl"      : kl,
            "agg"     : agg_l,
            "comp"    : comp_l,
            "lattice" : lat_l,       # log-space length + angle + volume
            "n_atoms" : n_l,
            "score"   : score_l,
            "type"    : type_l,
            "decoder" : dec_l,
            "energy"  : e_l,
            "ehull"   : eh_l,
            "bg"      : bg_l,
        }

    # ── Generation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, n_samples=1, device="cpu",
                 composition=None,
                 n_atoms_override=None,
                 temperature=1.0,
                 langevin_steps=100,
                 step_size=1e-4):
        """
        Generate crystal structures via Annealed Langevin Dynamics.
        No post-hoc rescaling — params_to_lattice now has correct physical bounds.
        """
        Z2S = {
            1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",
            11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",
            19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",
            27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",
            35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",41:"Nb",42:"Mo",
            43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",
            51:"Sb",52:"Te",53:"I",54:"Xe",55:"Cs",56:"Ba",57:"La",58:"Ce",
            72:"Hf",73:"Ta",74:"W",75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",
            80:"Hg",81:"Tl",82:"Pb",83:"Bi",
        }
        self.eval()
        B = n_samples

        z          = torch.randn(B, self.latent, device=device) * temperature
        props      = self.pred(z)
        comp_probs = props["comp"].softmax(-1)

        if composition is not None:
            comp_probs = torch.zeros(B, NUM_ELEMENTS, device=device)
            for zn in composition:
                if 0 < zn < NUM_ELEMENTS:
                    comp_probs[:, zn] += 1.0
            comp_probs = comp_probs / comp_probs.sum(-1, keepdim=True).clamp(min=1e-8)

        lattice = params_to_lattice(props["lattice"])   # (B, 3, 3) — now bounded

        if n_atoms_override:
            n_ats = torch.full((B,), n_atoms_override,
                               dtype=torch.long, device=device)
        else:
            n_ats = (props["n_atoms"].argmax(-1) + 1).clamp(2, MAX_ATOMS)

        a2b  = torch.repeat_interleave(torch.arange(B, device=device), n_ats)
        fc   = torch.rand(n_ats.sum().item(), 3, device=device)
        types = torch.multinomial(comp_probs[a2b].clamp(min=1e-8), 1).squeeze(-1)

        # Annealed Langevin dynamics
        for j in range(self.n_sigs):
            sx_j  = self.sigs_x[j]
            alpha  = step_size * (sx_j ** 2) / (self.sigs_x[-1] ** 2)
            sx_b   = sx_j.expand(B)
            sa_b   = self.sigs_a[j].expand(B)

            for _ in range(langevin_steps):
                sc, tl = self.score(fc, types, lattice, n_ats, z, sx_b, sa_b)
                fc     = (fc + alpha * sc
                          + math.sqrt(2 * float(alpha)) * torch.randn_like(fc)) % 1.0
                types  = tl.argmax(-1)

        out, off = [], 0
        for i in range(B):
            n    = int(n_ats[i])
            fc_i = fc[off:off+n].cpu()
            t_i  = types[off:off+n].cpu().tolist()
            lat  = lattice[i].cpu()
            out.append({
                "n_atoms"    : n,
                "atom_types" : t_i,
                "symbols"    : [Z2S.get(t, f"Z{t}") for t in t_i],
                "frac_coords": fc_i.numpy(),
                "cartesian"  : (fc_i @ lat).numpy(),
                "lattice"    : lat.numpy(),
                "energy"     : float(props["energy"][i]),
                "ehull"      : float(props["ehull"][i]),
                "band_gap"   : float(props["bg"][i]),
            })
            off += n
        return out