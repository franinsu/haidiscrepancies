from __future__ import annotations

import ast
import itertools
import random
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .common import (
    MAX_MULTI_SOLUTIONS,
    MIN_MULTI_SOLUTIONS,
    Puzzle,
    counter_to_regular_dict,
    solution_count_in_range,
    stable_hash,
)
from .probes import annotate_puzzle


TARGET = Fraction(24, 1)
CLASSIC_OVERUSED = {
    (1, 3, 4, 6),
    (1, 5, 5, 5),
    (3, 3, 8, 8),
    (4, 4, 10, 10),
    (5, 5, 5, 1),
}


@dataclass(frozen=True)
class ExprInfo:
    value: Fraction
    expr: str
    canon: str
    canon_key: Tuple[Any, ...]
    cost: float
    uses_division: bool
    has_fraction_intermediate: bool
    shape: str
    height: int
    ops: Tuple[str, ...]


def solve_numbers(numbers: Sequence[int]) -> List[Dict[str, Any]]:
    nums = tuple(numbers)
    n = len(nums)

    @lru_cache(maxsize=None)
    def build(mask: int) -> Tuple[ExprInfo, ...]:
        if mask and mask & (mask - 1) == 0:
            idx = mask.bit_length() - 1
            v = Fraction(nums[idx], 1)
            s = str(nums[idx])
            return (ExprInfo(v, s, s, ("n", nums[idx]), 0.0, False, False, "L", 0, ()),)

        seen: Dict[Tuple[Fraction, Tuple[Any, ...]], ExprInfo] = {}
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub < other:
                for left in build(sub):
                    for right in build(other):
                        for expr in itertools.chain(_combine(left, right), _combine(right, left)):
                            key = (expr.value, expr.canon_key)
                            if key not in seen or expr.cost < seen[key].cost:
                                seen[key] = expr
            sub = (sub - 1) & mask
        return tuple(seen.values())

    full = (1 << n) - 1
    solutions: Dict[str, ExprInfo] = {}
    for expr in build(full):
        if expr.value == TARGET:
            current = solutions.get(expr.canon)
            if current is None or expr.cost < current.cost:
                solutions[expr.canon] = expr
    ordered = sorted(solutions.values(), key=lambda e: (round(e.cost, 3), e.canon))
    return [_solution_record(e) for e in ordered]


def _combine(left: ExprInfo, right: ExprInfo) -> Iterable[ExprInfo]:
    specs = [
        ("+", left.value + right.value, 1.0),
        ("*", left.value * right.value, 1.2),
        ("-", left.value - right.value, 1.5),
    ]
    if right.value != 0:
        specs.append(("/", left.value / right.value, 2.0))

    for op, value, op_cost in specs:
        canon_key = _canonical_key(op, left.canon_key, right.canon_key)
        canon = _key_to_string(canon_key)
        shape = _canonical_shape(op, left.shape, right.shape)
        expr = canon
        unbalanced = abs(left.height - right.height) > 1
        frac_here = value.denominator != 1
        cost = (
            left.cost
            + right.cost
            + op_cost
            + (1.0 if frac_here else 0.0)
            + (0.5 if unbalanced else 0.0)
        )
        yield ExprInfo(
            value=value,
            expr=expr,
            canon=canon,
            canon_key=canon_key,
            cost=cost,
            uses_division=left.uses_division or right.uses_division or op == "/",
            has_fraction_intermediate=left.has_fraction_intermediate
            or right.has_fraction_intermediate
            or frac_here,
            shape=shape,
            height=max(left.height, right.height) + 1,
            ops=tuple(sorted(left.ops + right.ops + (op,))),
        )


def _canonical_key(op: str, left: Tuple[Any, ...], right: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Canonical key for common arithmetic equivalences.

    This handles associative/commutative chains for addition and
    multiplication, and also folds subtraction into signed additive terms
    and division into numerator/denominator factors. It intentionally does
    not apply distribution or general symbolic simplification.
    """
    if op == "+":
        lpos, lneg = _sum_parts(left)
        rpos, rneg = _sum_parts(right)
        return _make_sum_key(lpos + rpos, lneg + rneg)
    if op == "-":
        lpos, lneg = _sum_parts(left)
        rpos, rneg = _sum_parts(right)
        return _make_sum_key(lpos + rneg, lneg + rpos)
    if op == "*":
        lnum, lden = _product_parts(left)
        rnum, rden = _product_parts(right)
        return _make_product_key(lnum + rnum, lden + rden)
    if op == "/":
        lnum, lden = _product_parts(left)
        rnum, rden = _product_parts(right)
        return _make_product_key(lnum + rden, lden + rnum)
    raise ValueError(f"Unknown op {op}")


def _sum_parts(key: Tuple[Any, ...]) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]]]:
    if key and key[0] == "sum":
        return list(key[1]), list(key[2])
    return [key], []


def _product_parts(key: Tuple[Any, ...]) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]]]:
    if key and key[0] == "prod":
        return list(key[1]), list(key[2])
    return [key], []


def _make_sum_key(
    positives: Sequence[Tuple[Any, ...]], negatives: Sequence[Tuple[Any, ...]]
) -> Tuple[Any, ...]:
    pos = tuple(sorted(positives, key=_key_sort_string))
    neg = tuple(sorted(negatives, key=_key_sort_string))
    if len(pos) == 1 and not neg:
        return pos[0]
    return ("sum", pos, neg)


def _make_product_key(
    numerators: Sequence[Tuple[Any, ...]], denominators: Sequence[Tuple[Any, ...]]
) -> Tuple[Any, ...]:
    num = tuple(sorted(numerators, key=_key_sort_string))
    den = tuple(sorted(denominators, key=_key_sort_string))
    if len(num) == 1 and not den:
        return num[0]
    return ("prod", num, den)


def _key_sort_string(key: Tuple[Any, ...]) -> str:
    return _key_to_string(key)


def _key_to_string(key: Tuple[Any, ...]) -> str:
    tag = key[0]
    if tag == "n":
        return str(key[1])
    if tag == "sum":
        positives, negatives = key[1], key[2]
        pieces = [_key_to_string(child) for child in positives]
        if not pieces:
            pieces = ["0"]
        expr = "+".join(pieces)
        for child in negatives:
            expr += "-" + _key_to_string(child)
        return f"({expr})"
    if tag == "prod":
        numerators, denominators = key[1], key[2]
        num = "*".join(_key_to_string(child) for child in numerators) or "1"
        if not denominators:
            return f"({num})"
        den = "*".join(_key_to_string(child) for child in denominators)
        if len(denominators) > 1:
            den = f"({den})"
        return f"({num}/{den})"
    left, right = key[1], key[2]
    return f"({_key_to_string(left)}{tag}{_key_to_string(right)})"


def _canonical_shape(op: str, left: str, right: str) -> str:
    if op in {"+", "-"}:
        return _signed_shape(op, left, right)
    if op in {"*", "/"}:
        return _factored_shape(op, left, right)
    raise ValueError(f"Unknown op {op}")


def _signed_shape(op: str, left: str, right: str) -> str:
    lpos, lneg = _shape_sum_parts(left)
    rpos, rneg = _shape_sum_parts(right)
    if op == "+":
        pos, neg = lpos + rpos, lneg + rneg
    else:
        pos, neg = lpos + rneg, lneg + rpos
    return f"(sum:{','.join(sorted(pos))}|neg:{','.join(sorted(neg))})"


def _factored_shape(op: str, left: str, right: str) -> str:
    lnum, lden = _shape_product_parts(left)
    rnum, rden = _shape_product_parts(right)
    if op == "*":
        num, den = lnum + rnum, lden + rden
    else:
        num, den = lnum + rden, lden + rnum
    return f"(prod:{','.join(sorted(num))}|den:{','.join(sorted(den))})"


def _shape_sum_parts(shape: str) -> Tuple[List[str], List[str]]:
    if shape.startswith("(sum:") and "|neg:" in shape and shape.endswith(")"):
        body = shape[5:-1]
        pos_part, neg_part = body.split("|neg:", 1)
        return _split_shape_operands(pos_part), _split_shape_operands(neg_part)
    return [shape], []


def _shape_product_parts(shape: str) -> Tuple[List[str], List[str]]:
    if shape.startswith("(prod:") and "|den:" in shape and shape.endswith(")"):
        body = shape[6:-1]
        num_part, den_part = body.split("|den:", 1)
        return _split_shape_operands(num_part), _split_shape_operands(den_part)
    return [shape], []




def _split_shape_operands(inner: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
    parts.append(inner[start:])
    return [p for p in parts if p]


def _solution_record(expr: ExprInfo) -> Dict[str, Any]:
    return {
        "expression": expr.expr,
        "canonical_expression": expr.canon,
        "cost": round(expr.cost, 3),
        "uses_division": expr.uses_division,
        "has_fraction_intermediate": expr.has_fraction_intermediate,
        "tree_shape": expr.shape,
        "operation_pattern": "".join(expr.ops),
    }


def compute_features(numbers: Sequence[int], solutions: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    nums = tuple(sorted(numbers))
    costs = [float(s["cost"]) for s in solutions]
    op_counts = Counter(s["operation_pattern"] for s in solutions)
    shape_counts = Counter(s["tree_shape"] for s in solutions)
    has_direct = _has_direct_factor_pair(nums)
    features = {
        "num_solutions": len(solutions),
        "min_expression_cost": min(costs),
        "max_expression_cost": max(costs),
        "fraction_required": all(s["has_fraction_intermediate"] for s in solutions),
        "division_required": all(s["uses_division"] for s in solutions),
        "has_direct_factor_pair": has_direct,
        "num_repeated_values": sum(c - 1 for c in Counter(nums).values() if c > 1),
        "number_range": max(nums) - min(nums),
        "solution_tree_shape_counts": counter_to_regular_dict(shape_counts),
        "operation_pattern_counts": counter_to_regular_dict(op_counts),
        "canonical_hash": stable_hash({"numbers": nums}),
        "classic_overused": nums in CLASSIC_OVERUSED,
    }
    signature = {
        "number_multiset": list(nums),
        "repeat_pattern": tuple(sorted(Counter(nums).values(), reverse=True)),
        "operation_patterns": sorted(op_counts.keys())[:6],
        "any_division": any(s["uses_division"] for s in solutions),
        "any_subtraction": any("-" in s["operation_pattern"] for s in solutions),
        "tree_shape_family": sorted(shape_counts.keys())[0],
        "direct_factor_pair": has_direct,
        "classic_overused": nums in CLASSIC_OVERUSED,
    }
    bucket = assign_difficulty(features)
    return features, signature, bucket


def _has_direct_factor_pair(nums: Sequence[int]) -> bool:
    targets = {(3, 8), (4, 6), (2, 12), (1, 24)}
    for a, b in itertools.combinations(nums, 2):
        if tuple(sorted((a, b))) in targets:
            return True
    return False


def assign_difficulty(features: Dict[str, Any]) -> str:
    min_cost = features["min_expression_cost"]
    if min_cost <= 3.3 or (features["has_direct_factor_pair"] and min_cost <= 3.5):
        return "easy"
    if features["division_required"] and features["fraction_required"]:
        return "hard"
    if min_cost >= 4.3:
        return "hard"
    return "medium"


def make_puzzle(
    numbers: Sequence[int],
    min_solutions: int = MIN_MULTI_SOLUTIONS,
    max_solutions: int = MAX_MULTI_SOLUTIONS,
) -> Puzzle | None:
    solutions = solve_numbers(numbers)
    if not solution_count_in_range(len(solutions), min_solutions, max_solutions):
        return None
    features, signature, bucket = compute_features(numbers, solutions)
    nums = tuple(sorted(numbers))
    rendered = "Numbers: " + ", ".join(map(str, nums))
    prompt = (
        "Question {{ID}}:\n"
        "Use each of the numbers below exactly once, together with +, -, *, / and parentheses, to make 24.\n\n"
        f"{rendered}\n\nGive one valid expression."
    )
    return annotate_puzzle({
        "id": "",
        "puzzle_type": "arithmetic24",
        "prompt_text": prompt,
        "rendered_puzzle": rendered,
        "machine_readable_instance": {"numbers": list(nums), "target": 24},
        "num_solutions": len(solutions),
        "solutions": solutions,
        "features": features,
        "difficulty_bucket": bucket,
        "diversity_signature": signature,
    })


def generate_pool(seed: int, valid_target: int = 500, min_attempts: int = 5000) -> List[Puzzle]:
    rng = random.Random(seed)
    multisets = list(itertools.combinations_with_replacement(range(1, 14), 4))
    rng.shuffle(multisets)
    pool: List[Puzzle] = []
    attempts = 0
    for nums in itertools.cycle(multisets):
        attempts += 1
        if attempts > len(multisets):
            break
        p = make_puzzle(nums)
        if p is not None:
            pool.append(p)
        if len(pool) >= valid_target and attempts >= min_attempts:
            break
    return pool


def validate_expression(expression: str, numbers: Sequence[int], target: int = 24) -> bool:
    try:
        tree = ast.parse(expression, mode="eval")
        used: List[int] = []
        value = _eval_ast(tree.body, used)
    except Exception:
        return False
    return value == Fraction(target, 1) and sorted(used) == sorted(numbers)


def _eval_ast(node: ast.AST, used: List[int]) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        used.append(int(node.value))
        return Fraction(int(node.value), 1)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, used)
        right = _eval_ast(node.right, used)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError
            return left / right
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")
