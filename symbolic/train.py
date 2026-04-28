"""
Training script for symbolic JEPA world model.

Follows LeWM's lejepa_forward pattern: encode → predict → L2 loss + SIGReg.
Adds value head for cnf_distance prediction (planning guidance).

Usage:
    python3 -u -m symbolic.train --size large --seed 42 --max-epochs 20
"""

import os
import sys
import json
import time
import copy
import random
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from symbolic.build import build_model, CONFIGS
from symbolic.data import (
    load_cached_split, tree_to_tensor_data, find_all_rule_applications,
    cnf_distance, RuleID, Var, Not, And, Or, generate_random_expr,
    VARIABLES, NUM_RULES, MAX_NODES,
)

from torch.utils.data import Dataset, DataLoader


# ── Dataset (adapted from research/src/training.py) ────────────────────────

class RewriteDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "expr_node_types": torch.tensor(s.expr_tensor["node_types"], dtype=torch.long),
            "expr_var_ids": torch.tensor(s.expr_tensor["var_ids"], dtype=torch.long),
            "expr_adjacency": torch.tensor(s.expr_tensor["adjacency"], dtype=torch.long),
            "expr_num_nodes": torch.tensor(s.expr_tensor["num_nodes"], dtype=torch.long),
            "result_node_types": torch.tensor(s.result_tensor["node_types"], dtype=torch.long),
            "result_var_ids": torch.tensor(s.result_tensor["var_ids"], dtype=torch.long),
            "result_adjacency": torch.tensor(s.result_tensor["adjacency"], dtype=torch.long),
            "result_num_nodes": torch.tensor(s.result_tensor["num_nodes"], dtype=torch.long),
            "rule_id": torch.tensor(s.rule_id, dtype=torch.long),
            "result_cnf_dist": torch.tensor(cnf_distance(s.result), dtype=torch.float),
        }


def collate_batch(batch):
    result = {}
    for key in batch[0]:
        result[key] = torch.stack([b[key] for b in batch])
    return result


class ValueHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent):
        return F.softplus(self.net(latent))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_solve_rate(model, encoder, value_head, device,
                        num_problems=300, max_steps=25, seed=9999):
    rng = random.Random(seed)
    test_exprs = []
    for _ in range(num_problems * 3):
        e = generate_random_expr(6, rng=rng)
        if e.depth() >= 2 and cnf_distance(e) > 0:
            test_exprs.append(e)
        if len(test_exprs) >= num_problems:
            break

    model.eval(); encoder.eval()
    if value_head: value_head.eval()

    model_solved = oracle_solved = random_solved = 0
    depth_stats = defaultdict(lambda: {"total": 0, "solved": 0})

    for idx, expr in enumerate(test_exprs):
        if idx % 50 == 0 and idx > 0:
            print(f"    [{idx}/{len(test_exprs)}] solve: {model_solved/idx*100:.1f}%")

        depth = expr.depth()
        depth_stats[depth]["total"] += 1

        # Model-guided
        current = expr
        for step in range(max_steps):
            if cnf_distance(current) == 0:
                model_solved += 1; depth_stats[depth]["solved"] += 1; break
            apps = find_all_rule_applications(current)
            if not apps: break
            with torch.no_grad():
                td = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in
                      tree_to_tensor_data(current).items()
                      if k in ("node_types", "var_ids", "adjacency", "num_nodes")}
                cur_emb = encoder(td).last_hidden_state[:, 0]
                best_s, best_i = float("inf"), 0
                for i, (rid, _, _) in enumerate(apps):
                    rt = torch.tensor([[int(rid)]], device=device)
                    ae = model.action_encoder(rt)
                    pred = model.predict(cur_emb.unsqueeze(1), ae)[:, -1, :]
                    s = value_head(pred).item() if value_head else float("inf")
                    if s < best_s: best_s, best_i = s, i
            current = apps[best_i][2]

        # Oracle
        current = expr
        for step in range(max_steps):
            if cnf_distance(current) == 0: oracle_solved += 1; break
            apps = find_all_rule_applications(current)
            if not apps: break
            scores = [(cnf_distance(r) + 0.01 * r.node_count(), i) for i, (_, _, r) in enumerate(apps)]
            current = apps[min(scores)[1]][2]

        # Random
        current = expr; rng_r = random.Random(42)
        for step in range(max_steps):
            if cnf_distance(current) == 0: random_solved += 1; break
            apps = find_all_rule_applications(current)
            if not apps: break
            current = rng_r.choice(apps)[2]

    n = len(test_exprs)
    return {
        "solve_rate": model_solved / n, "oracle_rate": oracle_solved / n,
        "random_rate": random_solved / n, "num_problems": n,
        "depth_stats": {d: dict(s) for d, s in sorted(depth_stats.items())},
    }


# ── Training ────────────────────────────────────────────────────────────────

def train(size, seed, max_epochs=20, patience=5, min_epochs=8,
          value_weight=1.0, batch_size=None, data_dir=None):

    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    model, sigreg, cfg = build_model(size)
    model = model.to(device)
    sigreg = sigreg.to(device)
    encoder = model.encoder
    dim = cfg["dim"]
    lam = cfg["sigreg_lambda"]

    value_head = ValueHead(dim, hidden_dim=dim).to(device)

    model_p = sum(p.numel() for p in model.parameters())
    value_p = sum(p.numel() for p in value_head.parameters())
    total_p = model_p + value_p

    print(f"\n{'='*60}")
    print(f"LeWM SYMBOLIC — {size.upper()} ({total_p:,} params)")
    print(f"  Model: {model_p:,}  Value head: {value_p:,}")
    print(f"  Encoder: TreeEncoder ({cfg['enc_layers']} layers, dim={dim})")
    print(f"  Predictor: LeWM ARPredictor ({cfg['pred_depth']} layers, AdaLN-zero)")
    print(f"  SIGReg: LeWM official (lambda={lam})")
    print(f"  Loss: L2 (matching LeWM)")
    print(f"  Seed: {seed}")
    print(f"{'='*60}\n")

    assert total_p <= 50_000_000, f"Over budget: {total_p:,}"

    if batch_size is None:
        batch_size = 48 if size == "large" else 32

    train_samples = load_cached_split("train", data_dir)
    print(f"  Data: {len(train_samples):,} samples, batch={batch_size}")

    train_loader = DataLoader(RewriteDataset(train_samples), batch_size=batch_size,
                              shuffle=True, collate_fn=collate_batch, drop_last=True, num_workers=0)

    all_params = list(model.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience_cnt = 0
    best_model_state = best_value_state = None
    training_log = []
    start_time = time.time()

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train(); value_head.train()
        ep = {"pred": [], "sig": [], "val_h": []}

        for batch in train_loader:
            optimizer.zero_grad()

            # Encode (both paths get gradients — LeJEPA, no stop-grad)
            expr_td = {k: batch[f"expr_{k}"].to(device) for k in ["node_types", "var_ids", "adjacency", "num_nodes"]}
            result_td = {k: batch[f"result_{k}"].to(device) for k in ["node_types", "var_ids", "adjacency", "num_nodes"]}

            expr_emb = encoder(expr_td).last_hidden_state[:, 0]    # (B, D)
            result_emb = encoder(result_td).last_hidden_state[:, 0]  # (B, D)

            # Predict (LeWM pattern: predictor takes (emb, act_emb))
            rule_ids = batch["rule_id"].to(device).unsqueeze(1)  # (B, 1)
            act_emb = model.action_encoder(rule_ids)  # (B, 1, D)
            pred_emb = model.predict(expr_emb.unsqueeze(1), act_emb)[:, -1, :]  # (B, D)

            # L2 prediction loss (LeWM: .pow(2).mean())
            pred_loss = (pred_emb - result_emb).pow(2).mean()

            # SIGReg (LeWM's exact implementation)
            all_emb = torch.cat([expr_emb, result_emb], dim=0).unsqueeze(0)  # (1, 2B, D)
            sig_loss = sigreg(all_emb)

            # Value head (cnf_distance, dense signal)
            pred_cnf = value_head(result_emb.detach()).squeeze(-1)
            target_cnf = batch["result_cnf_dist"].to(device)
            val_loss = F.mse_loss(pred_cnf, target_cnf)

            total = pred_loss + lam * sig_loss + value_weight * val_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            ep["pred"].append(pred_loss.item())
            ep["sig"].append(sig_loss.item())
            ep["val_h"].append(val_loss.item())

        scheduler.step()

        # Free training memory before validation
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        # Validate (smaller batch to avoid OOM after training epoch)
        model.eval(); value_head.eval()
        if not hasattr(train, '_val_samples'):
            train._val_samples = load_cached_split("val", data_dir)
        val_loader = DataLoader(RewriteDataset(train._val_samples), batch_size=max(batch_size // 2, 16),
                                shuffle=False, collate_fn=collate_batch, drop_last=False, num_workers=0)
        val_losses = []
        val_embs = []
        with torch.no_grad():
            for vb in val_loader:
                etd = {k: vb[f"expr_{k}"].to(device) for k in ["node_types", "var_ids", "adjacency", "num_nodes"]}
                rtd = {k: vb[f"result_{k}"].to(device) for k in ["node_types", "var_ids", "adjacency", "num_nodes"]}
                e_emb = encoder(etd).last_hidden_state[:, 0]
                r_emb = encoder(rtd).last_hidden_state[:, 0]
                ri = vb["rule_id"].to(device).unsqueeze(1)
                ae = model.action_encoder(ri)
                pred = model.predict(e_emb.unsqueeze(1), ae)[:, -1, :]
                val_losses.append((pred - r_emb).pow(2).mean().item())
                if len(val_embs) < 10:
                    val_embs.append(e_emb)

        # Rank
        all_v = torch.cat(val_embs, dim=0).detach().cpu().float()
        fro = (all_v - all_v.mean(dim=0)).norm().item()
        if fro > 1e-6:
            try:
                S = torch.linalg.svdvals(all_v - all_v.mean(dim=0))
                rank_ratio = (S > S.max() * 0.01).sum().item() / dim
            except Exception:
                rank_ratio = 0.0
        else:
            rank_ratio = 0.0

        mean_val = np.mean(val_losses)
        epoch_time = time.time() - t0

        entry = {"epoch": epoch, "pred": np.mean(ep["pred"]), "sig": np.mean(ep["sig"]),
                 "val_h": np.mean(ep["val_h"]), "val": mean_val, "rank": rank_ratio, "time": epoch_time}
        training_log.append(entry)

        print(f"  Epoch {epoch:3d} | pred={entry['pred']:.6f} sig={entry['sig']:.4f} "
              f"val_h={entry['val_h']:.3f} val={mean_val:.6f} rank={rank_ratio:.2f} time={epoch_time:.0f}s")

        if mean_val < best_val:
            best_val = mean_val; patience_cnt = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_value_state = copy.deepcopy(value_head.state_dict())
            torch.save({"model": best_model_state, "value_head": best_value_state,
                        "epoch": epoch, "val": best_val, "seed": seed, "size": size},
                       ckpt_dir / f"lewm_{size}_seed{seed}_best.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= patience and epoch >= min_epochs:
                print(f"  Early stopping at epoch {epoch}"); break

    total_time = time.time() - start_time
    model.load_state_dict(best_model_state)
    value_head.load_state_dict(best_value_state)

    with open(log_dir / f"lewm_{size}_seed{seed}_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Solve rate
    print(f"\n  Evaluating SOLVE RATE...")
    solve = evaluate_solve_rate(model, encoder, value_head, device, 300, 25, 9999)

    print(f"\n{'='*60}")
    print(f"  LeWM {size.upper()} RESULTS ({total_p:,} params)")
    print(f"{'='*60}")
    print(f"  >>> SOLVE RATE:  {solve['solve_rate']*100:.1f}% <<<")
    print(f"      Oracle:      {solve['oracle_rate']*100:.1f}%")
    print(f"      Random:      {solve['random_rate']*100:.1f}%")
    print(f"      Rank:        {rank_ratio:.2f}")
    print(f"      Train time:  {total_time:.0f}s ({total_time/60:.1f}min)")
    if solve["depth_stats"]:
        print(f"\n  Per-depth:")
        for d, s in sorted(solve["depth_stats"].items()):
            if s["total"] > 0:
                print(f"    Depth {d}: {s['solved']}/{s['total']} = {s['solved']/s['total']*100:.0f}%")

    result = {"size": size, "params": total_p, "seed": seed, **solve,
              "rank": rank_ratio, "train_time": total_time}
    with open(log_dir / f"lewm_{size}_seed{seed}_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    train(args.size, args.seed, max_epochs=args.max_epochs, batch_size=args.batch_size)
