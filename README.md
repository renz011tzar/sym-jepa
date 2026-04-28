# Sym-JEPA: Latent World Model for Symbolic Rewrite Dynamics

A latent world model that learns the dynamics of propositional logic rewrite rules from data, without being given the rules explicitly. The model predicts expression states in latent space and uses a learned value function to plan simplification sequences toward Conjunctive Normal Form (CNF).

Uses SIGReg and ARPredictor components from [LeWorldModel](https://github.com/lucas-maes/le-wm) (Maes et al., 2026).

## Results

| Method | Solve Rate |
|--------|-----------|
| **Sym-JEPA** | **34.0%** |
| Greedy heuristic | 31.3% |
| Random | 18.0% |

31.4M parameters. Trained for 20 epochs.

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
