"""
Data generation for propositional logic rewrite dynamics.

Generates (expression, rule_id, position, result_expression) triples for training
a JEPA-style world model. All expressions are represented as binary trees.
"""

import random
import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from copy import deepcopy

import yaml
import numpy as np


# ── Node types ──────────────────────────────────────────────────────────────

class Op(IntEnum):
    """Operators in propositional logic."""
    VAR = 0       # leaf: variable
    TRUE = 1      # leaf: constant True
    FALSE = 2     # leaf: constant False
    NOT = 3       # unary
    AND = 4       # binary
    OR = 5        # binary
    IMPLIES = 6   # binary
    IFF = 7       # binary

    def is_leaf(self) -> bool:
        return self in (Op.VAR, Op.TRUE, Op.FALSE)

    def is_unary(self) -> bool:
        return self == Op.NOT

    def is_binary(self) -> bool:
        return self in (Op.AND, Op.OR, Op.IMPLIES, Op.IFF)


VARIABLES = ["p", "q", "r", "s"]
NUM_NODE_TYPES = len(Op)  # 8 types
NUM_RULES = 15


# ── Expression tree ─────────────────────────────────────────────────────────

@dataclass
class Expr:
    """A node in a propositional logic expression tree."""
    op: Op
    var_name: Optional[str] = None   # only for Op.VAR
    left: Optional['Expr'] = None
    right: Optional['Expr'] = None

    def depth(self) -> int:
        if self.op.is_leaf():
            return 0
        if self.op.is_unary():
            return 1 + self.left.depth()
        return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self) -> int:
        if self.op.is_leaf():
            return 1
        if self.op.is_unary():
            return 1 + self.left.node_count()
        return 1 + self.left.node_count() + self.right.node_count()

    def to_str(self) -> str:
        if self.op == Op.VAR:
            return self.var_name
        if self.op == Op.TRUE:
            return "T"
        if self.op == Op.FALSE:
            return "F"
        if self.op == Op.NOT:
            return f"(~{self.left.to_str()})"
        names = {Op.AND: "&", Op.OR: "|", Op.IMPLIES: "->", Op.IFF: "<->"}
        return f"({self.left.to_str()} {names[self.op]} {self.right.to_str()})"

    def __repr__(self):
        return self.to_str()

    def __eq__(self, other):
        if not isinstance(other, Expr):
            return False
        if self.op != other.op:
            return False
        if self.op == Op.VAR:
            return self.var_name == other.var_name
        if self.op.is_leaf():
            return True
        if self.op.is_unary():
            return self.left == other.left
        return self.left == other.left and self.right == other.right

    def __hash__(self):
        if self.op == Op.VAR:
            return hash((self.op, self.var_name))
        if self.op.is_leaf():
            return hash(self.op)
        if self.op.is_unary():
            return hash((self.op, hash(self.left)))
        return hash((self.op, hash(self.left), hash(self.right)))

    def evaluate(self, assignment: dict) -> bool:
        """Evaluate expression under a truth assignment."""
        if self.op == Op.VAR:
            return assignment[self.var_name]
        if self.op == Op.TRUE:
            return True
        if self.op == Op.FALSE:
            return False
        if self.op == Op.NOT:
            return not self.left.evaluate(assignment)
        l = self.left.evaluate(assignment)
        r = self.right.evaluate(assignment)
        if self.op == Op.AND:
            return l and r
        if self.op == Op.OR:
            return l or r
        if self.op == Op.IMPLIES:
            return (not l) or r
        if self.op == Op.IFF:
            return l == r
        raise ValueError(f"Unknown op: {self.op}")

    def truth_table(self, variables=None) -> tuple:
        """Compute truth table as a tuple of bools (canonical semantic fingerprint)."""
        if variables is None:
            variables = VARIABLES
        results = []
        for bits in range(2 ** len(variables)):
            assignment = {}
            for i, v in enumerate(variables):
                assignment[v] = bool((bits >> i) & 1)
            results.append(self.evaluate(assignment))
        return tuple(results)


# ── Helper constructors ─────────────────────────────────────────────────────

def Var(name: str) -> Expr:
    return Expr(Op.VAR, var_name=name)

def Not(e: Expr) -> Expr:
    return Expr(Op.NOT, left=e)

def And(a: Expr, b: Expr) -> Expr:
    return Expr(Op.AND, left=a, right=b)

def Or(a: Expr, b: Expr) -> Expr:
    return Expr(Op.OR, left=a, right=b)

def Implies(a: Expr, b: Expr) -> Expr:
    return Expr(Op.IMPLIES, left=a, right=b)

def Iff(a: Expr, b: Expr) -> Expr:
    return Expr(Op.IFF, left=a, right=b)

def TrueExpr() -> Expr:
    return Expr(Op.TRUE)

def FalseExpr() -> Expr:
    return Expr(Op.FALSE)


# ── Random expression generation ────────────────────────────────────────────

def generate_random_expr(
    max_depth: int,
    variables: list[str] = VARIABLES,
    rng: random.Random = None,
    operator_weights: dict = None,
) -> Expr:
    """Generate a random propositional logic expression of bounded depth."""
    if rng is None:
        rng = random.Random()
    if operator_weights is None:
        operator_weights = {"AND": 1.0, "OR": 1.0, "NOT": 0.5, "IMPLIES": 0.3, "IFF": 0.2}

    def _gen(depth_remaining: int) -> Expr:
        # At depth 0, must produce a leaf
        if depth_remaining <= 0:
            return _gen_leaf()

        # Weighted choice between leaf and operator
        # Bias toward operators at higher depths to avoid trivially shallow trees
        leaf_prob = 0.15 if depth_remaining > 1 else 0.3
        if rng.random() < leaf_prob:
            return _gen_leaf()

        # Choose operator
        ops = []
        weights = []
        for op_name, w in operator_weights.items():
            ops.append(Op[op_name])
            weights.append(w)
        total = sum(weights)
        weights = [w / total for w in weights]
        op = rng.choices(ops, weights=weights, k=1)[0]

        if op == Op.NOT:
            return Not(_gen(depth_remaining - 1))
        else:
            left = _gen(depth_remaining - 1)
            right = _gen(depth_remaining - 1)
            return Expr(op, left=left, right=right)

    def _gen_leaf() -> Expr:
        # Small chance of TRUE/FALSE constants
        if rng.random() < 0.1:
            return TrueExpr() if rng.random() < 0.5 else FalseExpr()
        return Var(rng.choice(variables))

    return _gen(max_depth)


# ── Rewrite rules ───────────────────────────────────────────────────────────

class RuleID(IntEnum):
    """The 15 rewrite rules for propositional logic toward CNF."""
    IMPL_ELIM = 0
    BICOND_ELIM = 1
    DOUBLE_NEG = 2
    DEMORGAN_AND = 3
    DEMORGAN_OR = 4
    NEG_TRUE = 5
    NEG_FALSE = 6
    DIST_OR_AND = 7
    COMPLEMENT_AND = 8
    COMPLEMENT_OR = 9
    IDENTITY_AND = 10
    IDENTITY_OR = 11
    ANNIHILATE_AND = 12
    ANNIHILATE_OR = 13
    IDEMPOTENT = 14


RULE_NAMES = [r.name for r in RuleID]


def _exprs_equal_commutative(a: Expr, b: Expr) -> bool:
    """Check if a == NOT(b) or b == NOT(a), for complement rules."""
    if a.op == Op.NOT and a.left == b:
        return True
    if b.op == Op.NOT and b.left == a:
        return True
    return False


def try_apply_rule(expr: Expr, rule: RuleID) -> Optional[Expr]:
    """Try to apply a rewrite rule at the ROOT of the expression.

    Returns the rewritten expression if the rule matches, else None.
    Pattern matching is commutative for AND/OR where applicable.
    """
    if rule == RuleID.IMPL_ELIM:
        # A IMPLIES B -> (NOT A) OR B
        if expr.op == Op.IMPLIES:
            return Or(Not(deepcopy(expr.left)), deepcopy(expr.right))

    elif rule == RuleID.BICOND_ELIM:
        # A IFF B -> ((NOT A) OR B) AND ((NOT B) OR A)
        if expr.op == Op.IFF:
            a, b = deepcopy(expr.left), deepcopy(expr.right)
            return And(Or(Not(deepcopy(a)), deepcopy(b)),
                       Or(Not(deepcopy(b)), deepcopy(a)))

    elif rule == RuleID.DOUBLE_NEG:
        # NOT (NOT A) -> A
        if expr.op == Op.NOT and expr.left.op == Op.NOT:
            return deepcopy(expr.left.left)

    elif rule == RuleID.DEMORGAN_AND:
        # NOT (A AND B) -> (NOT A) OR (NOT B)
        if expr.op == Op.NOT and expr.left.op == Op.AND:
            return Or(Not(deepcopy(expr.left.left)), Not(deepcopy(expr.left.right)))

    elif rule == RuleID.DEMORGAN_OR:
        # NOT (A OR B) -> (NOT A) AND (NOT B)
        if expr.op == Op.NOT and expr.left.op == Op.OR:
            return And(Not(deepcopy(expr.left.left)), Not(deepcopy(expr.left.right)))

    elif rule == RuleID.NEG_TRUE:
        # NOT TRUE -> FALSE
        if expr.op == Op.NOT and expr.left.op == Op.TRUE:
            return FalseExpr()

    elif rule == RuleID.NEG_FALSE:
        # NOT FALSE -> TRUE
        if expr.op == Op.NOT and expr.left.op == Op.FALSE:
            return TrueExpr()

    elif rule == RuleID.DIST_OR_AND:
        # A OR (B AND C) -> (A OR B) AND (A OR C)
        # Also: (B AND C) OR A -> (B OR A) AND (C OR A)
        if expr.op == Op.OR:
            if expr.right.op == Op.AND:
                a = deepcopy(expr.left)
                b = deepcopy(expr.right.left)
                c = deepcopy(expr.right.right)
                return And(Or(deepcopy(a), b), Or(a, c))
            if expr.left.op == Op.AND:
                a = deepcopy(expr.right)
                b = deepcopy(expr.left.left)
                c = deepcopy(expr.left.right)
                return And(Or(b, deepcopy(a)), Or(c, a))

    elif rule == RuleID.COMPLEMENT_AND:
        # A AND (NOT A) -> FALSE  (or (NOT A) AND A)
        if expr.op == Op.AND:
            if _exprs_equal_commutative(expr.left, expr.right):
                return FalseExpr()

    elif rule == RuleID.COMPLEMENT_OR:
        # A OR (NOT A) -> TRUE  (or (NOT A) OR A)
        if expr.op == Op.OR:
            if _exprs_equal_commutative(expr.left, expr.right):
                return TrueExpr()

    elif rule == RuleID.IDENTITY_AND:
        # A AND TRUE -> A  (or TRUE AND A)
        if expr.op == Op.AND:
            if expr.right.op == Op.TRUE:
                return deepcopy(expr.left)
            if expr.left.op == Op.TRUE:
                return deepcopy(expr.right)

    elif rule == RuleID.IDENTITY_OR:
        # A OR FALSE -> A  (or FALSE OR A)
        if expr.op == Op.OR:
            if expr.right.op == Op.FALSE:
                return deepcopy(expr.left)
            if expr.left.op == Op.FALSE:
                return deepcopy(expr.right)

    elif rule == RuleID.ANNIHILATE_AND:
        # A AND FALSE -> FALSE  (or FALSE AND A)
        if expr.op == Op.AND:
            if expr.right.op == Op.FALSE or expr.left.op == Op.FALSE:
                return FalseExpr()

    elif rule == RuleID.ANNIHILATE_OR:
        # A OR TRUE -> TRUE  (or TRUE OR A)
        if expr.op == Op.OR:
            if expr.right.op == Op.TRUE or expr.left.op == Op.TRUE:
                return TrueExpr()

    elif rule == RuleID.IDEMPOTENT:
        # A AND A -> A  or  A OR A -> A
        if expr.op in (Op.AND, Op.OR):
            if expr.left == expr.right:
                return deepcopy(expr.left)

    return None


# ── Position-aware rule application ─────────────────────────────────────────

def get_all_positions(expr: Expr) -> list[tuple[int, ...]]:
    """Return all positions (paths from root) in the expression tree.

    A position is a tuple of ints: () = root, (0,) = left child of root,
    (1,) = right child, (0, 1) = right child of left child, etc.
    """
    positions = [()]
    if expr.op.is_leaf():
        return positions
    if expr.left is not None:
        for p in get_all_positions(expr.left):
            positions.append((0,) + p)
    if expr.right is not None:
        for p in get_all_positions(expr.right):
            positions.append((1,) + p)
    return positions


def get_subtree(expr: Expr, position: tuple[int, ...]) -> Optional[Expr]:
    """Get the subtree at a given position."""
    node = expr
    for step in position:
        if step == 0:
            node = node.left
        elif step == 1:
            node = node.right
        else:
            return None
        if node is None:
            return None
    return node


def replace_subtree(expr: Expr, position: tuple[int, ...], new_subtree: Expr) -> Expr:
    """Return a new expression with the subtree at position replaced."""
    if len(position) == 0:
        return new_subtree
    expr_copy = deepcopy(expr)
    node = expr_copy
    for step in position[:-1]:
        if step == 0:
            node = node.left
        else:
            node = node.right
    if position[-1] == 0:
        node.left = new_subtree
    else:
        node.right = new_subtree
    return expr_copy


def find_all_rule_applications(expr: Expr) -> list[tuple[RuleID, tuple[int, ...], Expr]]:
    """Find all valid (rule, position, result) triples for a given expression."""
    applications = []
    positions = get_all_positions(expr)
    for pos in positions:
        subtree = get_subtree(expr, pos)
        if subtree is None:
            continue
        for rule in RuleID:
            result = try_apply_rule(subtree, rule)
            if result is not None:
                # Check the rewrite doesn't produce the same expression
                full_result = replace_subtree(expr, pos, result)
                if full_result != expr:
                    applications.append((rule, pos, full_result))
    return applications


# ── Reward signal ───────────────────────────────────────────────────────────

def cnf_distance(expr: Expr) -> int:
    """Count CNF structural violations in the expression."""
    violations = 0

    def _count(node: Expr, parent_op: Optional[Op] = None):
        nonlocal violations
        if node.op.is_leaf():
            return
        # IMPLIES or IFF present
        if node.op in (Op.IMPLIES, Op.IFF):
            violations += 1
        # NOT applied to non-variable/non-constant
        if node.op == Op.NOT and not node.left.op.is_leaf():
            violations += 1
        # OR with AND descendant (non-CNF nesting)
        if node.op == Op.OR and parent_op == Op.AND:
            pass  # This is fine in CNF: AND of ORs
        if node.op == Op.AND and parent_op == Op.OR:
            violations += 1  # OR of ANDs — need distribution
        # Recurse
        if node.left is not None:
            _count(node.left, node.op)
        if node.right is not None:
            _count(node.right, node.op)

    _count(expr)
    return violations


def compute_reward(expr: Expr, alpha: float = 1.0, beta: float = 0.01) -> float:
    """Compute the reward signal for a given expression state."""
    return -alpha * cnf_distance(expr) - beta * expr.node_count()


# ── Tree serialization (for model input) ────────────────────────────────────

MAX_NODES = 127  # 2^7 - 1, max nodes in depth-6 binary tree


def tree_to_tensor_data(expr: Expr) -> dict:
    """Convert expression tree to a dict of arrays for batched model input.

    Uses a level-order (BFS) encoding into fixed-size arrays.

    Returns:
        node_types: int array [MAX_NODES], node type IDs (0=padding)
        var_ids: int array [MAX_NODES], variable index (0-3 for p-s, -1 otherwise)
        adjacency: int array [MAX_NODES, 2], children indices (-1 = no child)
        num_nodes: int, actual number of nodes
        depth: int, tree depth
    """
    node_types = np.zeros(MAX_NODES, dtype=np.int64)
    var_ids = np.full(MAX_NODES, -1, dtype=np.int64)
    adjacency = np.full((MAX_NODES, 2), -1, dtype=np.int64)

    # BFS traversal
    queue = [(expr, 0)]
    idx = 0
    while queue and idx < MAX_NODES:
        node, pos = queue.pop(0)
        # node_type: shift by 1 so 0 = padding
        node_types[pos] = int(node.op) + 1
        if node.op == Op.VAR:
            var_ids[pos] = VARIABLES.index(node.var_name)
        if node.left is not None:
            child_idx = 2 * pos + 1
            if child_idx < MAX_NODES:
                adjacency[pos, 0] = child_idx
                queue.append((node.left, child_idx))
        if node.right is not None:
            child_idx = 2 * pos + 2
            if child_idx < MAX_NODES:
                adjacency[pos, 1] = child_idx
                queue.append((node.right, child_idx))
        idx += 1

    return {
        "node_types": node_types,
        "var_ids": var_ids,
        "adjacency": adjacency,
        "num_nodes": expr.node_count(),
        "depth": expr.depth(),
    }


# ── Dataset generation ──────────────────────────────────────────────────────

@dataclass
class RewriteSample:
    """A single (expression, rule, position, result) sample."""
    expr: Expr
    rule_id: int
    position: tuple
    result: Expr
    expr_tensor: dict = field(default_factory=dict)
    result_tensor: dict = field(default_factory=dict)


def generate_dataset(
    num_expressions: int,
    max_depth: int = 6,
    min_depth: int = 2,
    variables: list[str] = VARIABLES,
    operator_weights: dict = None,
    seed: int = 42,
) -> list[RewriteSample]:
    """Generate a dataset of rewrite samples.

    Generates num_expressions unique expressions and enumerates all valid
    rule applications for each.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    seen_hashes = set()
    samples = []
    expressions_generated = 0
    attempts = 0
    max_attempts = num_expressions * 20  # avoid infinite loops

    while expressions_generated < num_expressions and attempts < max_attempts:
        attempts += 1
        # Sample depth uniformly
        depth = rng.randint(min_depth, max_depth)
        expr = generate_random_expr(depth, variables, rng, operator_weights)

        # Ensure minimum depth
        if expr.depth() < min_depth:
            continue

        # Deduplicate by structural hash
        expr_hash = hash(expr)
        if expr_hash in seen_hashes:
            continue
        seen_hashes.add(expr_hash)

        # Find all valid rule applications
        applications = find_all_rule_applications(expr)
        if not applications:
            continue  # Skip expressions with no valid rewrites

        expr_tensor = tree_to_tensor_data(expr)
        for rule_id, position, result in applications:
            result_tensor = tree_to_tensor_data(result)
            samples.append(RewriteSample(
                expr=expr,
                rule_id=int(rule_id),
                position=position,
                result=result,
                expr_tensor=expr_tensor,
                result_tensor=result_tensor,
            ))

        expressions_generated += 1

    return samples


def generate_and_cache_splits(config_path: str = "research/config.yaml") -> dict:
    """Generate train/val/test splits and cache to disk."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ds_cfg = config["dataset"]
    cache_dir = ds_cfg["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    splits = {}
    # Use different seed offsets for each split to ensure no overlap
    split_configs = [
        ("train", ds_cfg["train_expressions"], 1000),
        ("val", ds_cfg["val_expressions"], 2000),
        ("test", ds_cfg["test_expressions"], 3000),
    ]

    for split_name, num_expr, seed_offset in split_configs:
        cache_file = os.path.join(cache_dir, f"{split_name}.pkl")
        print(f"Generating {split_name} split ({num_expr} expressions)...")
        samples = generate_dataset(
            num_expressions=num_expr,
            max_depth=ds_cfg["max_depth"],
            min_depth=ds_cfg["min_depth"],
            operator_weights=ds_cfg.get("operator_weights"),
            seed=seed_offset,
        )
        # Save
        with open(cache_file, "wb") as f:
            pickle.dump(samples, f)
        splits[split_name] = samples
        print(f"  {split_name}: {len(samples)} samples from {num_expr} target expressions")

    return splits


def load_cached_split(split_name: str, cache_dir: str = "research/data") -> list[RewriteSample]:
    """Load a cached dataset split from disk."""
    cache_file = os.path.join(cache_dir, f"{split_name}.pkl")
    with open(cache_file, "rb") as f:
        return pickle.load(f)


# ── Dataset statistics ──────────────────────────────────────────────────────

def compute_dataset_stats(samples: list[RewriteSample]) -> dict:
    """Compute statistics about a dataset split."""
    if not samples:
        return {"num_samples": 0}

    unique_exprs = set()
    depths = []
    node_counts = []
    rule_counts = {r.name: 0 for r in RuleID}
    result_depths = []
    result_node_counts = []

    for s in samples:
        unique_exprs.add(hash(s.expr))
        depths.append(s.expr.depth())
        node_counts.append(s.expr.node_count())
        rule_counts[RuleID(s.rule_id).name] += 1
        result_depths.append(s.result.depth())
        result_node_counts.append(s.result.node_count())

    return {
        "num_samples": len(samples),
        "num_unique_expressions": len(unique_exprs),
        "depth_mean": np.mean(depths),
        "depth_std": np.std(depths),
        "depth_min": min(depths),
        "depth_max": max(depths),
        "depth_distribution": {d: depths.count(d) for d in range(max(depths) + 1)},
        "node_count_mean": np.mean(node_counts),
        "node_count_std": np.std(node_counts),
        "rule_distribution": rule_counts,
        "result_depth_mean": np.mean(result_depths),
        "result_node_count_mean": np.mean(result_node_counts),
        "size_change_mean": np.mean([rc - nc for rc, nc in zip(result_node_counts, node_counts)]),
    }


if __name__ == "__main__":
    splits = generate_and_cache_splits()
    for name, samples in splits.items():
        stats = compute_dataset_stats(samples)
        print(f"\n=== {name} ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
