"""
Check training progress and optionally evaluate solve rate on latest checkpoint.

Usage:
    cd research/reference/le-wm
    python3 -m symbolic.check_progress --size large --seed 42
    python3 -m symbolic.check_progress --size large --seed 42 --solve-rate
"""

import os
import sys
import json
import argparse

import torch


from symbolic.build import build_model
from symbolic.train import ValueHead, evaluate_solve_rate, set_seed


def check(size, seed, run_solve_rate=False):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root, "checkpoints")
    log_dir = os.path.join(root, "logs")

    # Training log
    log_path = os.path.join(log_dir, f"lewm_{size}_seed{seed}_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        print(f"{'Epoch':>5} {'Pred':>10} {'SIGReg':>8} {'ValHead':>8} {'Val':>10} {'Rank':>6} {'Time':>6}")
        print("-" * 60)
        for e in log:
            print(f"{e['epoch']:>5} {e['pred']:>10.6f} {e['sig']:>8.4f} "
                  f"{e['val_h']:>8.3f} {e['val']:>10.6f} {e['rank']:>6.2f} {e['time']:>5.0f}s")
        print(f"\nEpochs completed: {len(log)}")
    else:
        print("No training log yet.")

    # Checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"lewm_{size}_seed{seed}_best.pt")
    if not os.path.exists(ckpt_path):
        print("No checkpoint yet.")
        return

    if run_solve_rate:
        print(f"\nEvaluating solve rate from checkpoint...")
        set_seed(seed)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        model, sigreg, cfg = build_model(size)
        model = model.to(device)
        value_head = ValueHead(cfg["dim"], hidden_dim=cfg["dim"]).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        value_head.load_state_dict(ckpt["value_head"])

        solve = evaluate_solve_rate(model, model.encoder, value_head, device, 200, 25, 9999)

        print(f"\n>>> SOLVE RATE: {solve['solve_rate']*100:.1f}% <<<")
        print(f"    Oracle:     {solve['oracle_rate']*100:.1f}%")
        print(f"    Random:     {solve['random_rate']*100:.1f}%")
        if solve["depth_stats"]:
            for d, s in sorted(solve["depth_stats"].items()):
                if s["total"] > 0:
                    print(f"    Depth {d}: {s['solved']}/{s['total']} = {s['solved']/s['total']*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solve-rate", action="store_true")
    args = parser.parse_args()
    check(args.size, args.seed, args.solve_rate)
