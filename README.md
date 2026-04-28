# Sym-JEPA: Latent World Model for Symbolic Rewrite Dynamics

A latent world model that learns the dynamics of propositional logic rewrite rules purely from data, the model is never given the rules explicitly. It observes (expression, rule_id, result) triples, learns to predict expression transformations in latent space, and uses a learned value function to plan simplification sequences toward Conjunctive Normal Form (CNF).

Uses SIGReg and ARPredictor components from [LeWorldModel](https://github.com/lucas-maes/le-wm) (Maes et al., 2026).

## Motivation

Symbolic simplification is typically solved by hardcoded algorithms or LLMs that already know the rules. We ask a different question: **can a neural world model learn rewrite dynamics from observation alone, and use those learned dynamics to plan?**

The model receives expression trees as structured input and a rule ID as an integer. It never sees the rule definitions. It learns what each rule does by observing before/after pairs, then uses that knowledge to select actions during planning.

## Results

Evaluated on 300 randomly generated non-CNF expressions (seed 9999, depth 2–6, 4 variables). Each method selects which rewrite rule to apply at each step; a symbolic engine executes the rewrite. Success = reaching CNF within 25 steps.

| Method | Solve Rate | How it picks actions |
|--------|-----------|----------------------|
| **Sym-JEPA** | **34.0%** | Learned value function scores predicted latent states |
| Greedy oracle | 31.3% | Sees the actual outcome of every action, picks lowest cnf_distance |
| Random | 18.0% | Uniform random selection |

31.4M parameters. Trained for 20 epochs.

**Sym-JEPA outperforms the greedy oracle despite having strictly less information.** The greedy oracle cheats — it computes the real cnf_distance of every candidate result before choosing. Our model never sees the actual outcome; it scores actions using predicted latent states from the world model. The greedy oracle is trapped by local optima (e.g., the distribution rule reduces cnf_distance by 1 but adds 20 nodes, creating a dead end). The learned value function avoids these traps because it was trained on 280K expression transitions and develops implicit lookahead.

## Architecture

```
Expression tree ──► TreeEncoder (8-layer Transformer) ──► latent
                                                            │
Rule ID ──► RuleEncoder ──► AdaLN conditioning             │
                                │                           │
                    ARPredictor (6-layer causal Transformer) │
                                │                           │
                         predicted next latent              │
                                │                           │
                         ValueHead ──► cnf_distance estimate
```

## Domain

### Propositional Logic

Expressions over variables {p, q, r, s} with operators {AND, OR, NOT, IMPLIES, IFF, TRUE, FALSE}. Expression trees bounded at depth ≤ 6 (max 127 nodes).

### 15 Rewrite Rules

| ID | Name | Pattern → Result |
|----|------|------------------|
| 0 | impl_elim | A → B ↦ ¬A ∨ B |
| 1 | bicond_elim | A ↔ B ↦ (¬A ∨ B) ∧ (¬B ∨ A) |
| 2 | double_neg | ¬¬A ↦ A |
| 3 | demorgan_and | ¬(A ∧ B) ↦ ¬A ∨ ¬B |
| 4 | demorgan_or | ¬(A ∨ B) ↦ ¬A ∧ ¬B |
| 5 | neg_true | ¬⊤ ↦ ⊥ |
| 6 | neg_false | ¬⊥ ↦ ⊤ |
| 7 | dist_or_and | A ∨ (B ∧ C) ↦ (A ∨ B) ∧ (A ∨ C) |
| 8 | complement_and | A ∧ ¬A ↦ ⊥ |
| 9 | complement_or | A ∨ ¬A ↦ ⊤ |
| 10 | identity_and | A ∧ ⊤ ↦ A |
| 11 | identity_or | A ∨ ⊥ ↦ A |
| 12 | annihilate_and | A ∧ ⊥ ↦ ⊥ |
| 13 | annihilate_or | A ∨ ⊤ ↦ ⊤ |
| 14 | idempotent | A ∧ A ↦ A, A ∨ A ↦ A |

Pattern matching is commutative for AND/OR (e.g., complement_and matches both A ∧ ¬A and ¬A ∧ A).

### Canonical Form (CNF)

An expression is in simplified CNF when:
- No IMPLIES or IFF operators remain
- NOT appears only directly on variables (negation normal form)
- Structure is AND of ORs: (l₁ ∨ l₂) ∧ (l₃ ∨ l₄) ∧ ...

### Reward Signal

`cnf_distance(expr)` counts structural violations of CNF form in the AST. O(n) computation, no external tools. Range: 0 (is CNF) to ~20+ (deeply nested non-CNF).

### Dataset

All data synthetically generated. Deterministic with fixed seeds.

| Split | Unique Expressions | Total Samples | Seed |
|-------|-------------------|---------------|------|
| Train | 50,000 | 280,286 | 1000 |
| Val | 5,000 | 27,095 | 2000 |
| Test | 10,000 | 53,766 | 3000 |

Each sample is an (expression, rule_id, position, result) tuple. ~5.5 valid rule applications per expression on average. No expression appears in multiple splits.

## Repository Structure

```
├── lewm/                      # Components from LeWorldModel (MIT license)
│   ├── jepa.py                # JEPA class: encode, predict, rollout, criterion
│   └── module.py              # SIGReg, ARPredictor, ConditionalBlock, Attention
├── symbolic/                  # Domain-specific code (ours)
│   ├── encoder.py             # TreeEncoder for expression trees
│   ├── rule_encoder.py        # Discrete rule embedding
│   ├── build.py               # Model assembly
│   ├── data.py                # Expression trees, 15 rewrite rules, dataset generation
│   ├── train.py               # Training + solve rate evaluation
│   └── check_progress.py      # Monitor training
├── requirements.txt
└── LICENSE
```

## Usage

```bash
pip install -r requirements.txt

# Generate data
python -c "
from symbolic.data import generate_dataset
import pickle, os
os.makedirs('data', exist_ok=True)
for name, n, seed in [('train', 50000, 1000), ('val', 5000, 2000), ('test', 10000, 3000)]:
    samples = generate_dataset(num_expressions=n, seed=seed)
    with open(f'data/{name}.pkl', 'wb') as f: pickle.dump(samples, f)
    print(f'{name}: {len(samples)} samples')
"

# Train
python -u -m symbolic.train --size large --seed 42 --max-epochs 20

# Check solve rate
python -m symbolic.check_progress --size large --seed 42 --solve-rate
```

## Acknowledgments

This project uses SIGReg and ARPredictor from [LeWorldModel](https://github.com/lucas-maes/le-wm) by Maes, Le Lidec, Scieur, LeCun & Balestriero (arXiv 2603.19312, 2026), licensed under MIT.

## License

MIT
