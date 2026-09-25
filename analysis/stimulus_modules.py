"""Numerical stimulus modules for the study.

Extracted from the reviewed study implementation; presentation is separate.
"""

from __future__ import annotations

import json


import random


from collections import Counter, defaultdict


from dataclasses import dataclass, field


from pathlib import Path


from typing import Any, Callable, Iterable


import numpy as np


MODULE_ORDER = ("cue", "spatial", "formulation", "pair", "transfer")


TYPE_ORDER = ("arithmetic24", "maze", "grid_placement", "minesweeper_lite", "mini_sudoku")


TYPE_LABELS = {
    "arithmetic24": "Arithmetic",
    "maze": "Maze",
    "grid_placement": "Rooks",
    "minesweeper_lite": "Minesweeper",
    "mini_sudoku": "Sudoku",
}


FAMILY_FIELD = {
    "cue": "module_family_id",
    "spatial": "spatial_id",
    "formulation": "formulation_id",
    "pair": "pair_id",
    "transfer": "transfer_id",
}


ROLE_BASELINE = "baseline"


ROLE_CUE_ARM = "cue_trial"


ROLES_A = frozenset({"A", "A_prime"})


ROLES_B = frozenset({"B", "B_alone", "B_target"})


CUE_BASELINE = "baseline_no_cue"


CUE_ARMS = ("cue_half_a", "cue_half_b")


SPATIAL_ORIGINAL = "original"


SPATIAL_REFLECTIONS = ("mirror_lr", "mirror_tb", "mirror_lr_tb")


FORMULATION_ORIGINAL = "original_order"


FORMULATION_PERTURBED = "shuffled_order"


PAIR_ALONE = "no_prime"


PAIR_RELATED = "related"


PAIR_CONTROL = "unrelated_control"


TRANSFER_ALONE = "no_prime"


TRANSFER_PRIME = "strategy_prime"


BOOTSTRAP_REPEATS = 5000


BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Trial:
    id: str
    module: str
    family_id: str
    puzzle_type: str
    condition: str
    role: str
    sequence_id: str | None
    sequence_position: int | None
    abstract_ids: tuple[str, ...]
    coord_of: dict[str, tuple[int, int]]
    target_ids: frozenset[str] | None
    decoy_coords: frozenset[tuple[int, int]] | None
    candidate_coords: frozenset[tuple[int, int]] | None
    related_map: dict[str, str] | None
    spatial_transform: str | None
    board: tuple[str, ...] | None
    numbers: tuple[int, ...] | None
    display_numbers: tuple[int, ...] | None
    strategy_signature: str | None
    num_solutions: int
    record: dict[str, Any] = field(compare=False, repr=False)

    @property
    def id_of_coord(self) -> dict[tuple[int, int], str]:
        return {coordinate: abstract_id for abstract_id, coordinate in self.coord_of.items()}

    @property
    def rows(self) -> int:
        return len(self.board) if self.board else 0

    @property
    def cols(self) -> int:
        return len(self.board[0]) if self.board else 0


@dataclass
class Family:
    module: str
    family_id: str
    puzzle_type: str
    trials: list[Trial]

    def by_condition(self, condition: str) -> Trial:
        matches = [trial for trial in self.trials if trial.condition == condition and trial.role not in ROLES_A]
        if len(matches) != 1:
            raise ValueError(f"{self.module} family {self.family_id}: expected one trial with condition {condition}, found {len(matches)}")
        return matches[0]

    def by_role(self, role: str) -> list[Trial]:
        return [trial for trial in self.trials if trial.role == role]

    def sequences(self) -> dict[str, dict[str, Trial]]:
        """Map condition -> {'A': trial or absent, 'B': trial} for sequence modules."""
        grouped: dict[str, dict[str, Trial]] = defaultdict(dict)
        for trial in self.trials:
            if trial.sequence_id is None:
                raise ValueError(f"{self.module} trial {trial.id} lacks a sequence_id")
            slot = "A" if trial.role in ROLES_A else "B" if trial.role in ROLES_B else None
            if slot is None:
                raise ValueError(f"{self.module} trial {trial.id} has unexpected role {trial.role}")
            if slot in grouped[trial.condition]:
                raise ValueError(f"{self.module} family {self.family_id}: duplicate {slot} trial in condition {trial.condition}")
            grouped[trial.condition][slot] = trial
        return dict(grouped)

    @property
    def abstract_ids(self) -> tuple[str, ...]:
        """Abstract ids of the measured (non-primer) trials, identical across them."""
        return next(trial for trial in self.trials if trial.role not in ROLES_A).abstract_ids


@dataclass
class Design:
    families: dict[str, list[Family]]  # module -> families in stratum order then first appearance
    trials: dict[str, Trial]

    def family_of(self, trial_id: str) -> Family:
        trial = self.trials[trial_id]
        for family in self.families[trial.module]:
            if family.family_id == trial.family_id:
                return family
        raise KeyError(trial_id)

    def strata(self, module: str) -> list[tuple[str, list[str]]]:
        grouped: dict[str, list[str]] = {}
        for family in self.families[module]:
            grouped.setdefault(family.puzzle_type, []).append(family.family_id)
        return [(puzzle_type, grouped[puzzle_type]) for puzzle_type in TYPE_ORDER if puzzle_type in grouped]

    def family_order(self, module: str) -> list[str]:
        return [family_id for _, ids in self.strata(module) for family_id in ids]

    def blocks(self) -> list[dict[str, Any]]:
        """Blocks implied by the metadata, for the cross-check against module_blocks.jsonl."""
        blocks: list[dict[str, Any]] = []
        for module in MODULE_ORDER:
            for family in self.families[module]:
                if module in ("pair", "transfer"):
                    for condition, slots in family.sequences().items():
                        trials = [slots[slot] for slot in ("A", "B") if slot in slots]
                        blocks.append({
                            "block_id": trials[0].sequence_id,
                            "module": module,
                            "puzzle_type": family.puzzle_type,
                            "trial_ids": frozenset(trial.id for trial in trials),
                            "condition": condition,
                        })
                else:
                    blocks.append({
                        "block_id": family.family_id,
                        "module": module,
                        "puzzle_type": family.puzzle_type,
                        "trial_ids": frozenset(trial.id for trial in family.trials),
                        "condition": None,
                    })
        return blocks


def _coordinate(value: Any) -> tuple[int, int]:
    return (int(value[0]), int(value[1]))


def _candidate_coordinates(puzzle_type: str, board: list[str] | None) -> frozenset[tuple[int, int]] | None:
    """Cells a wrong answer can land on: hidden Minesweeper cells or empty Sudoku cells."""
    if board is None:
        return None
    marker = {"minesweeper_lite": "?", "mini_sudoku": "."}.get(puzzle_type)
    if marker is None:
        return None
    return frozenset(
        (row + 1, column + 1)
        for row, line in enumerate(board)
        for column, value in enumerate(line)
        if value == marker
    )


def _trial_from_record(record: dict[str, Any]) -> Trial:
    meta = record["module_metadata"]
    module = str(meta["module"])
    instance = record.get("machine_readable_instance") or {}
    solutions = record["solutions"]
    abstract_ids = tuple(str(solution["abstract_solution_id"]) for solution in solutions)
    if len(set(abstract_ids)) != len(abstract_ids):
        raise ValueError(f"{record['id']}: duplicate abstract ids")
    coord_of = {
        str(solution["abstract_solution_id"]): _coordinate(solution["coordinate"])
        for solution in solutions
        if solution.get("coordinate") is not None
    }
    targets = meta.get("target_abstract_solution_ids")
    decoys = meta.get("highlighted_invalid_coordinates")
    related_map = meta.get("related_solution_map")
    if isinstance(related_map, str):
        related_map = json.loads(related_map)
    board = instance.get("board")
    numbers = instance.get("numbers")
    display_numbers = instance.get("display_numbers")
    return Trial(
        id=str(record["id"]),
        module=module,
        family_id=str(meta[FAMILY_FIELD[module]]),
        puzzle_type=str(record["puzzle_type"]),
        condition=str(meta["condition"]),
        role=str(meta["role"]),
        sequence_id=str(meta["sequence_id"]) if meta.get("sequence_id") else None,
        sequence_position=int(meta["sequence_position"]) if meta.get("sequence_position") is not None else None,
        abstract_ids=abstract_ids,
        coord_of=coord_of,
        target_ids=frozenset(str(value) for value in targets) if targets else None,
        decoy_coords=frozenset(_coordinate(value) for value in decoys) if decoys else None,
        candidate_coords=_candidate_coordinates(str(record["puzzle_type"]), board) if board else None,
        related_map={str(key): str(value) for key, value in related_map.items()} if related_map else None,
        spatial_transform=str(meta["spatial_transform"]) if meta.get("spatial_transform") else None,
        board=tuple(str(line) for line in board) if board else None,
        numbers=tuple(int(value) for value in numbers) if numbers else None,
        display_numbers=tuple(int(value) for value in display_numbers) if display_numbers else None,
        strategy_signature=str(meta.get("prime_strategy_signature") or meta.get("target_strategy_signature") or "") or None,
        num_solutions=int(record["num_solutions"]),
        record=record,
    )


def _reflect(coordinate: tuple[int, int], transform: str, rows: int, cols: int) -> tuple[int, int]:
    row, column = coordinate
    if transform in ("mirror_h", "mirror_hv"):
        column = cols - column + 1
    if transform in ("mirror_v", "mirror_hv"):
        row = rows - row + 1
    return (row, column)


def _check_family(family: Family) -> dict[str, Any]:
    """Assert the design invariants of one family and return recorded facts."""
    facts: dict[str, Any] = {}
    ids = set(family.abstract_ids)
    for trial in family.trials:
        if set(trial.abstract_ids) != ids and not (family.module in ("pair", "transfer") and trial.role in ROLES_A):
            raise ValueError(f"{family.module} {family.family_id}: abstract ids differ across trials ({trial.id})")
    if family.module == "cue":
        baselines = [trial for trial in family.trials if trial.role == ROLE_BASELINE]
        arms = [trial for trial in family.trials if trial.role == ROLE_CUE_ARM]
        if len(baselines) != 1 or len(arms) != len(CUE_ARMS) or baselines[0].condition != CUE_BASELINE:
            raise ValueError(f"cue {family.family_id}: expected one baseline and {len(CUE_ARMS)} arms")
        if {arm.condition for arm in arms} != set(CUE_ARMS):
            raise ValueError(f"cue {family.family_id}: unexpected arm conditions")
        targets = [arm.target_ids for arm in arms]
        if any(target is None for target in targets) or (targets[0] & targets[1]) or (targets[0] | targets[1]) != ids:
            raise ValueError(f"cue {family.family_id}: arm targets do not partition the valid set")
        for arm in arms:
            if arm.decoy_coords is None or len(arm.decoy_coords) != len(arm.target_ids):
                raise ValueError(f"cue {family.family_id}: decoy count differs from target count in {arm.id}")
            if arm.decoy_coords & set(arm.coord_of.values()):
                raise ValueError(f"cue {family.family_id}: a decoy is a valid cell in {arm.id}")
            if arm.candidate_coords is None or not arm.decoy_coords <= arm.candidate_coords:
                raise ValueError(f"cue {family.family_id}: decoys are not hidden or empty cells in {arm.id}")
        if arms[0].decoy_coords & arms[1].decoy_coords:
            raise ValueError(f"cue {family.family_id}: decoys shared across arms")
        facts["target_sizes"] = [len(arm.target_ids) for arm in sorted(arms, key=lambda t: CUE_ARMS.index(t.condition))]
    elif family.module == "spatial":
        original = family.by_condition(SPATIAL_ORIGINAL)
        reflections = [family.by_condition(condition) for condition in SPATIAL_REFLECTIONS]
        if original.spatial_transform not in (None, "none"):
            raise ValueError(f"spatial {family.family_id}: original trial has a transform")
        for trial in reflections:
            for abstract_id, coordinate in original.coord_of.items():
                if trial.coord_of[abstract_id] != _reflect(coordinate, trial.spatial_transform, original.rows, original.cols):
                    raise ValueError(f"spatial {family.family_id}: {trial.id} does not reflect {abstract_id}")
        facts["transforms"] = {trial.condition: trial.spatial_transform for trial in reflections}
    elif family.module == "formulation":
        original = family.by_condition(FORMULATION_ORIGINAL)
        perturbed = family.by_condition(FORMULATION_PERTURBED)
        expressions = {
            trial.id: {str(s["abstract_solution_id"]): str(s["canonical_expression"]) for s in trial.record["solutions"]}
            for trial in (original, perturbed)
        }
        if expressions[original.id] != expressions[perturbed.id]:
            raise ValueError(f"formulation {family.family_id}: expression classes differ")
        if sorted(original.numbers) != sorted(perturbed.numbers):
            raise ValueError(f"formulation {family.family_id}: number multisets differ")
        displayed = perturbed.display_numbers or perturbed.numbers
        if sorted(displayed) != sorted(original.numbers):
            raise ValueError(f"formulation {family.family_id}: displayed numbers are not a permutation")
        facts["original_display"] = list(original.display_numbers or original.numbers)
        facts["perturbed_display"] = list(displayed)
        facts["is_exact_reversal"] = list(displayed) == list(reversed(facts["original_display"]))
    elif family.module == "pair":
        sequences = family.sequences()
        if set(sequences) != {PAIR_ALONE, PAIR_RELATED, PAIR_CONTROL}:
            raise ValueError(f"pair {family.family_id}: unexpected conditions {sorted(sequences)}")
        b_trials = [sequences[condition]["B"] for condition in (PAIR_ALONE, PAIR_RELATED, PAIR_CONTROL)]
        if "A" in sequences[PAIR_ALONE] or "A" not in sequences[PAIR_RELATED] or "A" not in sequences[PAIR_CONTROL]:
            raise ValueError(f"pair {family.family_id}: sequence roles are inconsistent")
        if len({trial.board for trial in b_trials}) != 1:
            raise ValueError(f"pair {family.family_id}: B boards differ across conditions")
        related_b = sequences[PAIR_RELATED]["B"]
        related_a = sequences[PAIR_RELATED]["A"]
        if related_b.related_map is None:
            raise ValueError(f"pair {family.family_id}: related B lacks related_solution_map")
        if set(related_b.related_map) != set(related_a.abstract_ids) or set(related_b.related_map.values()) != ids:
            raise ValueError(f"pair {family.family_id}: related_solution_map is not a bijection from A ids onto B ids")
        control_a = sequences[PAIR_CONTROL]["A"]
        if set(control_a.abstract_ids) & ids:
            raise ValueError(f"pair {family.family_id}: control A shares abstract ids with B")
        facts["related_map_is_identity"] = all(key == value for key, value in related_b.related_map.items())
        facts["k_b"] = len(ids)
        facts["k_a_related"] = len(related_a.abstract_ids)
        facts["k_a_control"] = len(control_a.abstract_ids)
    elif family.module == "transfer":
        sequences = family.sequences()
        if set(sequences) != {TRANSFER_ALONE, TRANSFER_PRIME}:
            raise ValueError(f"transfer {family.family_id}: unexpected conditions {sorted(sequences)}")
        primer = sequences[TRANSFER_PRIME]["A"]
        b_prime = sequences[TRANSFER_PRIME]["B"]
        b_alone = sequences[TRANSFER_ALONE]["B"]
        if primer.num_solutions != 1:
            raise ValueError(f"transfer {family.family_id}: primer is not single-solution")
        if b_prime.board != b_alone.board or b_prime.target_ids != b_alone.target_ids or not b_prime.target_ids:
            raise ValueError(f"transfer {family.family_id}: B trials differ or lack a target")
        if primer.strategy_signature != b_prime.strategy_signature:
            raise ValueError(f"transfer {family.family_id}: primer and target strategy signatures differ")
        facts["strategy_signature"] = primer.strategy_signature
        facts["k_b"] = len(ids)
        facts["target_size"] = len(b_prime.target_ids)
    return facts


def load_design(project: Path) -> tuple[Design, dict[str, Any]]:
    """Build the module design from trial metadata and cross-check the block file."""
    with (project / "stimuli/modules/all_module_trials.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    trials: dict[str, Trial] = {}
    grouped: dict[tuple[str, str], list[Trial]] = {}
    first_seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        module = str(record["module_metadata"]["module"])
        if module not in FAMILY_FIELD:
            raise ValueError(f"unknown module {module} for trial {record['id']}")
        trial = _trial_from_record(record)
        trials[trial.id] = trial
        key = (module, trial.family_id)
        grouped.setdefault(key, []).append(trial)
        first_seen.setdefault(key, index)
    families: dict[str, list[Family]] = {module: [] for module in MODULE_ORDER}
    for (module, family_id), members in grouped.items():
        puzzle_types = {trial.puzzle_type for trial in members}
        if len(puzzle_types) != 1:
            raise ValueError(f"{module} family {family_id} spans puzzle types {sorted(puzzle_types)}")
        families[module].append(Family(module, family_id, puzzle_types.pop(), members))
    for module in MODULE_ORDER:
        families[module].sort(key=lambda family: (TYPE_ORDER.index(family.puzzle_type), first_seen[(module, family.family_id)]))
    design = Design(families=families, trials=trials)

    facts: dict[str, dict[str, Any]] = {module: {} for module in MODULE_ORDER}
    for module in MODULE_ORDER:
        for family in families[module]:
            facts[module][family.family_id] = _check_family(family)

    with (project / "stimuli/modules/module_blocks.jsonl").open(encoding="utf-8") as handle:
        block_file = [json.loads(line) for line in handle if line.strip()]
    expected = {
        (str(block["block_id"]), str(block["module"]), str(block["puzzle_type"]), frozenset(map(str, block["trial_ids"])), block.get("condition"))
        for block in block_file
    }
    derived = {(b["block_id"], b["module"], b["puzzle_type"], b["trial_ids"], b["condition"]) for b in design.blocks()}
    if expected != derived:
        raise ValueError("metadata-derived blocks do not reproduce module_blocks.jsonl")
    file_family_order = {module: [] for module in MODULE_ORDER}
    for block in block_file:
        family_id = design.trials[str(block["trial_ids"][0])].family_id
        if family_id not in file_family_order[block["module"]]:
            file_family_order[block["module"]].append(family_id)
    order_matches = {module: file_family_order[module] == design.family_order(module) for module in MODULE_ORDER}
    summary = {
        "families_per_module": {module: len(families[module]) for module in MODULE_ORDER},
        "strata": {module: {puzzle_type: len(ids) for puzzle_type, ids in design.strata(module)} for module in MODULE_ORDER},
        "block_file_reproduced": True,
        "family_order_matches_block_file": order_matches,
        "facts": facts,
    }
    return design, summary


@dataclass
class Answer:
    cls: str | None
    pair_key: tuple[Any, ...]


@dataclass
class SourceData:
    name: str
    answers: dict[str, list[Answer]]
    invalid_reasons: Counter[str] = field(default_factory=Counter)

    def valid(self, trial_id: str) -> list[Answer]:
        if trial_id not in self.answers:
            raise KeyError(f"{self.name}: no responses for {trial_id}")
        return [answer for answer in self.answers[trial_id] if answer.cls is not None]

    def counts(self, trial_id: str) -> Counter[str]:
        counter = Counter(answer.cls for answer in self.valid(trial_id))
        if not counter:
            raise ValueError(f"{self.name}: no counted responses for {trial_id}")
        return counter

    def all_rows(self, trial_id: str) -> list[Answer]:
        if trial_id not in self.answers:
            raise KeyError(f"{self.name}: no responses for {trial_id}")
        return self.answers[trial_id]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_human_answers(repro: Path, design: Design) -> SourceData:
    """Load final retained answers and participant/sequence pairing keys."""
    answers = defaultdict(list)
    seen = set()
    for row in read_jsonl(repro / 'human/module_retained.jsonl'):
        trial = design.trials.get(str(row['puzzle_id']))
        if trial is None:
            continue
        participant = row.get('participant_id')
        if not participant:
            raise ValueError(f'Human row for {trial.id} lacks participant_id')
        pair_key = ('human',str(participant),str(row.get('sequence_id') or ''))
        if trial.sequence_id is not None and not row.get('sequence_id'):
            raise ValueError(f'Missing sequence_id for {trial.id}')
        if (trial.id,pair_key) in seen:
            raise ValueError(f'Duplicate Human pair key for {trial.id}')
        seen.add((trial.id,pair_key))
        cls = str(row['abstract_solution_id']) if row.get('is_correct') and row.get('abstract_solution_id') else None
        answers[trial.id].append(Answer(cls=cls,pair_key=pair_key))
    return SourceData(name='Human',answers=dict(answers))


def load_model_answers(
    repro: Path,
    design: Design,
    providers: dict[str, dict[str, Any]],
    prompts: dict[str, str],
    condition_key: Callable[[str, str, str], str],
) -> dict[str, SourceData]:
    """Scored model module rows, one SourceData per provider effort/prompt cell."""
    sources: dict[str, SourceData] = {}
    for provider, provider_meta in providers.items():
        inverse_conditions = {value: key for key, value in provider_meta["conditions"].items()}
        per_source: dict[str, dict[str, list[Answer]]] = defaultdict(lambda: defaultdict(list))
        reasons: dict[str, Counter[str]] = defaultdict(Counter)
        for row in read_jsonl(repro / "ai" / f"{provider_meta['slug']}_module.jsonl"):
            effort = inverse_conditions.get(str(row.get("api_model_condition")))
            prompt = str(row.get("prompt_condition") or "")
            if effort != 'low' or prompt != 'direct_solve':
                continue
            trial = design.trials.get(str(row["puzzle_id"]))
            if trial is None:
                continue
            source = condition_key(provider, effort, prompt)
            if not row.get("request_id") or row.get("sample_index") is None:
                raise ValueError(f"model row for {trial.id} lacks request_id or sample_index")
            pair_key = ("api", str(row["request_id"]), int(row["sample_index"]))
            cls = str(row["abstract_solution_id"]) if row.get("is_valid") and row.get("abstract_solution_id") else None
            per_source[source][trial.id].append(Answer(cls=cls,pair_key=pair_key))
        for source, answers in per_source.items():
            sources[source] = SourceData(name=source, answers=dict(answers), invalid_reasons=reasons[source])
    return sources


def paired_classes(source: SourceData, a_trial: Trial, b_trial: Trial) -> list[tuple[str, str]]:
    a_by_key = {answer.pair_key: answer.cls for answer in source.valid(a_trial.id)}
    pairs = [(a_by_key[answer.pair_key], answer.cls) for answer in source.valid(b_trial.id) if answer.pair_key in a_by_key]
    if not pairs:
        raise ValueError(f"{source.name}: no paired answers for {a_trial.id} and {b_trial.id}")
    return pairs


def bootstrap_draws(stratum_sizes: tuple[int, ...], repeats: int = None, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Stratified block-index draws; the stream matches the established pipeline."""
    if repeats is None:
        repeats = BOOTSTRAP_REPEATS
    rng = random.Random(seed)
    offsets = np.cumsum((0, *stratum_sizes[:-1]))
    return np.asarray(
        [[int(offset) + rng.randrange(size) for offset, size in zip(offsets, stratum_sizes) for _ in range(size)] for _ in range(repeats)],
        dtype=int,
    )


def summarize_blocks(values_by_family: dict[str, float], strata: list[tuple[str, list[str]]], draws: np.ndarray) -> dict[str, Any]:
    ordered = [family_id for _, ids in strata for family_id in ids]
    if set(ordered) != set(values_by_family):
        raise ValueError("block values do not cover the design families")
    blocks = np.array([values_by_family[family_id] for family_id in ordered], dtype=float)
    means = blocks[draws].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {"estimate": float(blocks.mean()), "ci95": [float(low), float(high)], "n_blocks": len(ordered)}


def primer_trial_ids(design: Design) -> set[str]:
    """Single-solution transfer primers, identified by role rather than by id."""
    return {trial.id for trial in design.trials.values() if trial.role in ROLES_A and trial.module == "transfer"}
