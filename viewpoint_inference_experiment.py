#!/usr/bin/env python3
"""
Study 1: active-inference viewpoint model discovery using pymdp.

This is a Python replacement for the exploratory MATLAB simulation. It builds
synthetic topographic grid worlds in which food is partly hidden by terrain,
then compares active-inference agent families whose policy priors encode
different terrain-use hypotheses.

The central model is `efficient_epistemology`: policies are favoured when they
are expected to reduce uncertainty about food location per unit energetic cost.
Other candidate families encode cost-only, prospect, refuge, stationarity,
attainment, and combined mechanisms.

Run:
    python viewpoint_inference_experiment_fixed.py --test-mode
    python viewpoint_inference_experiment_fixed.py --output-dir outputs_viewpoint_pymdp_study1
    python viewpoint_inference_experiment_fixed.py --test-mode --no-progress

Required packages:
    pip install inferactively-pymdp tqdm

Notes:
    - pymdp is used for discrete-state inference and policy inference.
    - The environment dynamics, topography generation, line-of-sight calculation,
      model recovery, and output tables are implemented here so the experiment is
      fully reproducible and not tied to SPM/MATLAB.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional but recommended
    _tqdm = None


ACTION_NAMES = ["stay", "north", "south", "west", "east", "scan"]
ACTION_DELTAS = {
    "stay": (0, 0),
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
    "scan": (0, 0),
}
N_ACTIONS = len(ACTION_NAMES)
NONE_SEEN = -1


@dataclass(frozen=True)
class Config:
    output_dir: Path = Path("outputs_viewpoint_pymdp_study1")
    seed: int = 20260429
    grid_size: int = 9
    n_worlds_per_condition: int = 4
    n_agents: int = 20
    n_trials_per_agent: int = 8
    horizon: int = 18
    policy_len: int = 2
    view_radius: int = 5
    observer_height: float = 0.08
    los_detection_prob: float = 0.92
    los_false_positive_prob: float = 0.015
    food_here_detection_prob: float = 0.98
    start_energy: float = 1.0
    energy_min_success: float = 0.15
    base_metabolic_cost: float = 0.025
    move_cost_scale: float = 0.135
    scan_cost: float = 0.055
    stay_cost: float = 0.020
    food_gain: float = 0.45
    hazard_cost_scale: float = 0.08
    gamma: float = 2.5
    action_temperature: float = 1.0
    min_policy_prior: float = 1e-8
    ppc_replications: int = 8
    test_mode: bool = False
    progress: bool = True

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def json_dir(self) -> Path:
        return self.output_dir / "json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: dict[str, float]
    c_food_here: float = 4.5
    c_food_seen: float = 0.35
    c_viewpoint: float = 0.0
    gamma: float = 2.5
    policy_prior_temperature: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class WorldFamily:
    name: str
    manipulated_variable: str
    primary_outcome: str
    description: str
    levels: tuple[str, ...] = ("low", "medium", "high")


@dataclass
class GridWorld:
    id: str
    family: str
    level: str
    level_numeric: int
    size: int
    height: np.ndarray
    slope: np.ndarray
    transition_to: np.ndarray
    legal_action: np.ndarray
    movement_cost: np.ndarray
    los_visible: np.ndarray
    prospect: np.ndarray
    refuge: np.ndarray
    stationarity: np.ndarray
    attainment: np.ndarray
    resource: np.ndarray
    hazard: np.ndarray
    legibility: np.ndarray
    viewpoints: np.ndarray
    food_prior: np.ndarray
    start_cells: np.ndarray

    @property
    def n_cells(self) -> int:
        return self.size * self.size

    def rc_to_cell(self, r: int, c: int) -> int:
        return int(r * self.size + c)

    def cell_to_rc(self, idx: int) -> tuple[int, int]:
        return int(idx // self.size), int(idx % self.size)


@dataclass
class PymdpModel:
    A: list[Any]
    B: list[Any]
    C: list[Any]
    D: list[Any]
    A_dependencies: list[list[int]]
    B_dependencies: list[list[int]]
    control_fac_idx: list[int]


@dataclass
class TrialSimulation:
    trial_row: dict[str, Any]
    decision_rows: list[dict[str, Any]]


def require_pymdp() -> tuple[Any, Any, Any]:
    """Import the current JAX pymdp API lazily so syntax checks work without it."""
    try:
        from jax import numpy as jnp
        from jax import random as jr
        from pymdp.agent import Agent
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "This experiment requires the current JAX pymdp package. Install it with:\n"
            "    pip install inferactively-pymdp\n"
            "If you are using a locked environment, install it in a fresh virtualenv."
        ) from exc
    return jnp, jr, Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pymdp Study 1 viewpoint model-discovery simulation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_viewpoint_pymdp_study1"))
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--n-worlds-per-condition", type=int, default=4)
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--n-trials-per-agent", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--policy-len", type=int, default=2)
    parser.add_argument("--test-mode", action="store_true", help="Use a small fast run for debugging.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars and ETA display.")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    progress = not args.no_progress
    if args.test_mode:
        return Config(
            output_dir=args.output_dir,
            seed=args.seed,
            grid_size=min(args.grid_size, 6),
            n_worlds_per_condition=1,
            n_agents=1,
            n_trials_per_agent=1,
            horizon=min(args.horizon, 5),
            policy_len=min(args.policy_len, 1),
            ppc_replications=1,
            test_mode=True,
            progress=progress,
        )
    return Config(
        output_dir=args.output_dir,
        seed=args.seed,
        grid_size=args.grid_size,
        n_worlds_per_condition=args.n_worlds_per_condition,
        n_agents=args.n_agents,
        n_trials_per_agent=args.n_trials_per_agent,
        horizon=args.horizon,
        policy_len=args.policy_len,
        test_mode=False,
        progress=progress,
    )


def ensure_dirs(cfg: Config) -> None:
    for d in [cfg.output_dir, cfg.tables_dir, cfg.figures_dir, cfg.json_dir]:
        d.mkdir(parents=True, exist_ok=True)


class NullProgress:
    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix_str(self, s: str) -> None:
        return None

    def close(self) -> None:
        return None


def progress_bar(total: int, desc: str, unit: str, cfg: Config):
    if cfg.progress and _tqdm is not None:
        return _tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, smoothing=0.05)
    return NullProgress()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d} h {minutes:02d} min {sec:02d} s"
    return f"{minutes:d} min {sec:02d} s"


def active_levels(family: WorldFamily, cfg: Config) -> tuple[str, ...]:
    # Test mode keeps both low and high levels so contrast code is exercised,
    # but omits the medium level to keep the run short.
    return ("low", "high") if cfg.test_mode else family.levels


def make_agent_instance(Agent: Any, **kwargs: Any) -> Any:
    # Current pymdp stores generated policies inside an Equinox static module.
    # Passing NumPy arrays avoids avoidable caller-side static JAX arrays, but
    # pymdp/equinox may still emit the same warning while constructing the
    # internal policy container. Suppress only that exact internal warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"A JAX array is being set as static!.*",
            category=UserWarning,
        )
        return Agent(**kwargs)


def define_world_families() -> list[WorldFamily]:
    return [
        WorldFamily("prospect", "prospect", "viewpoint_choice_probability", "Higher line-of-sight prospect."),
        WorldFamily("refuge", "refuge", "dwell_time_at_viewpoint", "Greater sheltered/safe station availability."),
        WorldFamily("stationarity", "stationarity", "scan_choice_probability", "Flatter safer dwell sites for stationary sampling."),
        WorldFamily("attainment", "attainment", "viewpoint_choice_probability", "More prominent or elevated candidate viewpoints."),
        WorldFamily("complexity_information", "information_richness", "entropy_reduction", "More informative food-disclosing sightlines."),
        WorldFamily("complexity_legibility", "legibility", "entropy_reduction", "More or less interpretable visual scene structure."),
        WorldFamily("effort_movement_cost", "movement_cost", "viewpoint_choice_probability", "Higher energetic movement cost."),
        WorldFamily("effort_metabolic_pressure", "metabolic_pressure", "regulation_success", "Higher baseline metabolic pressure."),
        WorldFamily("effort_resource_abundance", "resource_abundance", "revisit_rate", "More abundant food resources."),
    ]


def specify_candidate_models() -> list[ModelSpec]:
    """Candidate active-inference policy-prior families.

    All models share a preference for eating. They differ in policy priors over
    candidate action sequences. The efficient epistemology model is the explicit
    test of information gain per energetic movement.
    """
    return [
        ModelSpec(
            "cost_only",
            {"movement_cost": -1.45, "hazard": -0.55},
            description="Avoid movement and hazard; no explicit viewpoint mechanism.",
        ),
        ModelSpec(
            "prospect_only",
            {"movement_cost": -0.65, "prospect": 1.45, "hazard": -0.30},
            c_viewpoint=0.10,
            description="Favours high line-of-sight positions.",
        ),
        ModelSpec(
            "refuge_only",
            {"movement_cost": -0.60, "refuge": 1.35, "hazard": -0.70},
            description="Favours sheltered and low-hazard positions.",
        ),
        ModelSpec(
            "stationarity_only",
            {"movement_cost": -0.70, "stationarity": 1.55, "hazard": -0.35},
            c_viewpoint=0.10,
            description="Favours places where stationary sampling is cheap and safe.",
        ),
        ModelSpec(
            "attainment_only",
            {"movement_cost": -0.80, "attainment": 1.45, "hazard": -0.25},
            description="Favours prominent/elevated places independent of information gain.",
        ),
        ModelSpec(
            "prospect_stationarity",
            {"movement_cost": -0.75, "prospect": 1.05, "stationarity": 1.05, "hazard": -0.35},
            c_viewpoint=0.15,
            description="Prospect plus dwell-compatible stationarity.",
        ),
        ModelSpec(
            "prospect_refuge",
            {"movement_cost": -0.75, "prospect": 1.05, "refuge": 1.00, "hazard": -0.60},
            c_viewpoint=0.15,
            description="Prospect plus refuge/shelter.",
        ),
        ModelSpec(
            "efficient_epistemology",
            {
                "movement_cost": -0.50,
                "efficient_information": 2.00,
                "stationarity": 0.45,
                "hazard": -0.45,
            },
            c_food_seen=0.15,
            c_viewpoint=0.20,
            description="Favours expected uncertainty reduction per unit movement/scan cost.",
        ),
        ModelSpec(
            "full_view_model",
            {
                "movement_cost": -0.85,
                "prospect": 0.75,
                "refuge": 0.55,
                "stationarity": 0.70,
                "attainment": 0.45,
                "resource": 0.40,
                "hazard": -0.55,
                "efficient_information": 1.15,
            },
            c_food_seen=0.25,
            c_viewpoint=0.20,
            description="Combined prospect/refuge/stationarity/attainment/energy model.",
        ),
    ]


def level_to_numeric(level: str) -> int:
    return {"low": -1, "medium": 0, "high": 1}[level]


def clamp01(x: np.ndarray | float) -> np.ndarray | float:
    return np.minimum(1.0, np.maximum(0.0, x))


def normalise01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x / max(float(temperature), 1e-8)
    x = x - np.nanmax(x)
    e = np.exp(x)
    s = np.sum(e)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(x) / max(1, x.size)
    return e / s


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def make_synthetic_height(size: int, rng: np.random.Generator, family: str, level_value: int) -> np.ndarray:
    x = np.linspace(-2.8, 2.8, size)
    y = np.linspace(-2.8, 2.8, size)
    X, Y = np.meshgrid(x, y)
    ridge = 1.2 * np.exp(-((X + 0.9) ** 2 / 0.8 + (Y - 0.2) ** 2 / 4.0))
    ridge += 1.0 * np.exp(-((X - 1.1) ** 2 / 1.1 + (Y + 0.7) ** 2 / 2.6))
    peak = 1.5 * np.exp(-((X - 0.1) ** 2 / 0.65 + (Y - 1.4) ** 2 / 0.65))
    basin = -0.8 * np.exp(-((X + 1.4) ** 2 / 0.9 + (Y + 1.2) ** 2 / 0.9))
    rolling = 0.25 * np.sin(1.4 * X) * np.cos(1.2 * Y)
    noise = 0.06 * rng.standard_normal((size, size))
    Z = ridge + peak + basin + rolling + noise
    if family == "prospect":
        Z += 0.20 * level_value * np.exp(-((X - 0.7) ** 2 / 0.9 + (Y - 0.9) ** 2 / 0.9))
    elif family == "attainment":
        Z += 0.25 * level_value * np.exp(-((X - 0.2) ** 2 / 0.6 + (Y - 1.2) ** 2 / 0.6))
    elif family == "effort_movement_cost":
        Z += 0.18 * level_value * np.sin(2.1 * X + 0.5) * np.sin(2.0 * Y)
    return normalise01(Z)


def compute_slope(height: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(height)
    return normalise01(np.hypot(gx, gy))


def line_of_sight(height: np.ndarray, a: tuple[int, int], b: tuple[int, int], observer_height: float) -> bool:
    ar, ac = a
    br, bc = b
    dr = br - ar
    dc = bc - ac
    n = max(abs(dr), abs(dc))
    if n <= 1:
        return True
    z0 = height[ar, ac] + observer_height
    z1 = height[br, bc]
    for step in range(1, n):
        t = step / n
        rr = int(round(ar + t * dr))
        cc = int(round(ac + t * dc))
        expected = z0 + t * (z1 - z0)
        if height[rr, cc] > expected + 0.015:
            return False
    return True


def build_los_matrix(height: np.ndarray, view_radius: int, observer_height: float) -> np.ndarray:
    size = height.shape[0]
    n = size * size
    los = np.zeros((n, n), dtype=bool)
    for i in range(n):
        r0, c0 = divmod(i, size)
        for j in range(n):
            r1, c1 = divmod(j, size)
            dist = math.hypot(r1 - r0, c1 - c0)
            if dist <= view_radius and line_of_sight(height, (r0, c0), (r1, c1), observer_height):
                los[i, j] = True
    return los


def build_transitions_and_costs(height: np.ndarray, slope: np.ndarray, family: str, level_value: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = height.shape[0]
    n = size * size
    transition_to = np.zeros((n, N_ACTIONS), dtype=int)
    legal = np.ones((n, N_ACTIONS), dtype=bool)
    costs = np.zeros((n, N_ACTIONS), dtype=float)
    movement_multiplier = 1.0 + (0.35 * level_value if family == "effort_movement_cost" else 0.0)
    for idx in range(n):
        r, c = divmod(idx, size)
        for a_idx, action in enumerate(ACTION_NAMES):
            dr, dc = ACTION_DELTAS[action]
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= size or cc < 0 or cc >= size:
                transition_to[idx, a_idx] = idx
                legal[idx, a_idx] = action in {"stay", "scan"}
                costs[idx, a_idx] = 0.15 if action not in {"stay", "scan"} else 0.02
                continue
            nxt = rr * size + cc
            transition_to[idx, a_idx] = nxt
            climb = max(0.0, float(height[rr, cc] - height[r, c]))
            if action == "scan":
                costs[idx, a_idx] = 0.09 + 0.08 * float(slope[r, c])
            elif action == "stay":
                costs[idx, a_idx] = 0.025
            else:
                costs[idx, a_idx] = movement_multiplier * (0.08 + 0.18 * climb + 0.10 * float(slope[rr, cc]))
    finite = costs[np.isfinite(costs)]
    costs = costs / max(float(np.nanmax(finite)), 1e-9)
    return transition_to, legal, costs


def smooth_grid_feature(values: np.ndarray, size: int, radius: int = 1) -> np.ndarray:
    grid = values.reshape(size, size)
    out = np.zeros_like(grid, dtype=float)
    for r in range(size):
        for c in range(size):
            r0, r1 = max(0, r - radius), min(size, r + radius + 1)
            c0, c1 = max(0, c - radius), min(size, c + radius + 1)
            out[r, c] = float(np.mean(grid[r0:r1, c0:c1]))
    return out.reshape(-1)


def build_grid_world(cfg: Config, family: WorldFamily, level: str, world_index: int, rng: np.random.Generator) -> GridWorld:
    level_value = level_to_numeric(level)
    size = cfg.grid_size
    n = size * size
    height = make_synthetic_height(size, rng, family.name, level_value)
    slope_grid = compute_slope(height)
    slope = slope_grid.reshape(-1)
    transition_to, legal_action, movement_cost = build_transitions_and_costs(height, slope_grid, family.name, level_value)
    los_visible = build_los_matrix(height, cfg.view_radius, cfg.observer_height)

    visible_count = normalise01(los_visible.sum(axis=1).astype(float))
    height_flat = height.reshape(-1)
    ruggedness = smooth_grid_feature(slope, size, radius=1)
    centrality = grid_centrality(size)

    prospect = normalise01(0.70 * visible_count + 0.30 * height_flat)
    refuge = normalise01(0.45 * (1.0 - slope) + 0.35 * (1.0 - centrality) + 0.20 * ruggedness)
    stationarity = normalise01(0.55 * (1.0 - slope) + 0.30 * refuge + 0.15 * (1.0 - ruggedness))
    attainment = normalise01(0.65 * height_flat + 0.35 * local_prominence(height).reshape(-1))
    resource = normalise01(0.45 * (1.0 - height_flat) + 0.35 * refuge + 0.20 * (1.0 - slope))
    hazard = normalise01(0.45 * slope + 0.35 * height_flat + 0.20 * (1.0 - refuge))
    legibility = normalise01(0.55 * prospect + 0.45 * (1.0 - ruggedness))

    # Controlled family manipulations on derived affordances. This keeps the
    # physical terrain plausible while creating interpretable one-factor sweeps.
    high_prospect_mask = prospect >= np.quantile(prospect, 0.72)
    if family.name == "prospect":
        prospect = clamp01(prospect + 0.16 * level_value * high_prospect_mask)
    elif family.name == "refuge":
        refuge = clamp01(refuge + 0.18 * level_value * (refuge >= np.quantile(refuge, 0.60)))
    elif family.name == "stationarity":
        stationarity = clamp01(stationarity + 0.18 * level_value * high_prospect_mask)
    elif family.name == "attainment":
        attainment = clamp01(attainment + 0.18 * level_value * high_prospect_mask)
    elif family.name == "complexity_information":
        prospect = clamp01(prospect + 0.10 * level_value * high_prospect_mask)
        legibility = clamp01(legibility + 0.08 * level_value)
    elif family.name == "complexity_legibility":
        legibility = clamp01(legibility + 0.18 * level_value)
    elif family.name == "effort_resource_abundance":
        resource = clamp01(resource + 0.18 * level_value)
    elif family.name == "effort_metabolic_pressure":
        hazard = clamp01(hazard + 0.10 * level_value)

    viewshed_quality = normalise01(0.45 * prospect + 0.25 * stationarity + 0.20 * legibility + 0.10 * attainment)
    viewpoints = viewshed_quality >= np.quantile(viewshed_quality, 0.82)

    food_prior = normalise01(0.65 * resource + 0.20 * refuge + 0.15 * (1.0 - hazard)) + 1e-6
    food_prior = food_prior / food_prior.sum()

    start_mask = np.zeros(n, dtype=bool)
    for c in range(size):
        idx = (size - 1) * size + c
        if height_flat[idx] <= np.quantile(height_flat, 0.55):
            start_mask[idx] = True
    start_cells = np.where(start_mask)[0]
    if start_cells.size == 0:
        start_cells = np.array([n - size], dtype=int)

    return GridWorld(
        id=f"{family.name}_{level}_{world_index:02d}",
        family=family.name,
        level=level,
        level_numeric=level_value,
        size=size,
        height=height,
        slope=slope,
        transition_to=transition_to,
        legal_action=legal_action,
        movement_cost=movement_cost,
        los_visible=los_visible,
        prospect=np.asarray(prospect, dtype=float),
        refuge=np.asarray(refuge, dtype=float),
        stationarity=np.asarray(stationarity, dtype=float),
        attainment=np.asarray(attainment, dtype=float),
        resource=np.asarray(resource, dtype=float),
        hazard=np.asarray(hazard, dtype=float),
        legibility=np.asarray(legibility, dtype=float),
        viewpoints=np.asarray(viewpoints, dtype=bool),
        food_prior=np.asarray(food_prior, dtype=float),
        start_cells=np.asarray(start_cells, dtype=int),
    )


def grid_centrality(size: int) -> np.ndarray:
    coords = np.array([(r, c) for r in range(size) for c in range(size)], dtype=float)
    centre = np.array([(size - 1) / 2, (size - 1) / 2], dtype=float)
    d = np.linalg.norm(coords - centre, axis=1)
    return normalise01(1.0 - d)


def local_prominence(height: np.ndarray) -> np.ndarray:
    size = height.shape[0]
    out = np.zeros_like(height, dtype=float)
    for r in range(size):
        for c in range(size):
            r0, r1 = max(0, r - 1), min(size, r + 2)
            c0, c1 = max(0, c - 1), min(size, c + 2)
            out[r, c] = height[r, c] - float(np.mean(height[r0:r1, c0:c1]))
    return normalise01(out)


def generate_world_bank(cfg: Config) -> list[GridWorld]:
    rng = np.random.default_rng(cfg.seed)
    worlds: list[GridWorld] = []
    jobs = [
        (family, level, w)
        for family in define_world_families()
        for level in active_levels(family, cfg)
        for w in range(cfg.n_worlds_per_condition)
    ]
    with progress_bar(len(jobs), "Generating worlds", "world", cfg) as pbar:
        for family, level, w in jobs:
            worlds.append(build_grid_world(cfg, family, level, w + 1, rng))
            pbar.update(1)
    return worlds


def make_pymdp_model(world: GridWorld, model: ModelSpec, D_location: np.ndarray, D_food: np.ndarray, policy_len: int = 1) -> tuple[Any, PymdpModel]:
    _, _, Agent = require_pymdp()
    n = world.n_cells

    # A[0]: exact location observation, depends on location factor only.
    A_location = np.eye(n, dtype=float)

    # A[1]: visual food-location observation. Observation categories are
    # 0..n-1 = food seen at that cell, n = no food seen.
    A_visual = np.zeros((n + 1, n, n), dtype=float)
    for loc in range(n):
        visible_foods = np.where(world.los_visible[loc])[0]
        for food in range(n):
            if world.los_visible[loc, food]:
                p = world.los_detection_prob if hasattr(world, "los_detection_prob") else 0.92
                A_visual[food, loc, food] = p
                A_visual[n, loc, food] = 1.0 - p
            else:
                A_visual[n, loc, food] = 1.0 - 0.015
                if visible_foods.size:
                    A_visual[visible_foods, loc, food] += 0.015 / visible_foods.size
                else:
                    A_visual[n, loc, food] = 1.0

    # A[2]: whether the agent is on the food cell.
    A_food_here = np.zeros((2, n, n), dtype=float)
    for loc in range(n):
        for food in range(n):
            if loc == food:
                A_food_here[1, loc, food] = 0.98
                A_food_here[0, loc, food] = 0.02
            else:
                A_food_here[1, loc, food] = 0.02
                A_food_here[0, loc, food] = 0.98

    # A[3]: whether current location is a viewpoint-quality location.
    A_viewpoint = np.zeros((2, n), dtype=float)
    A_viewpoint[1, world.viewpoints] = 0.98
    A_viewpoint[0, world.viewpoints] = 0.02
    A_viewpoint[1, ~world.viewpoints] = 0.02
    A_viewpoint[0, ~world.viewpoints] = 0.98

    # B[0]: controlled location transitions. B[next_loc, prev_loc, action]
    B_location = np.zeros((n, n, N_ACTIONS), dtype=float)
    for loc in range(n):
        for action in range(N_ACTIONS):
            nxt = int(world.transition_to[loc, action])
            B_location[nxt, loc, action] = 1.0

    # B[1]: static food-location factor. It is not controllable.
    B_food = np.zeros((n, n, 1), dtype=float)
    for food in range(n):
        B_food[food, food, 0] = 1.0

    C_location = np.zeros(n, dtype=float)
    C_visual = np.zeros(n + 1, dtype=float)
    C_visual[:n] = model.c_food_seen
    C_visual[n] = -0.05
    C_food_here = np.array([0.0, model.c_food_here], dtype=float)
    C_viewpoint = np.array([0.0, model.c_viewpoint], dtype=float)

    pymdp_model = PymdpModel(
        A=[A_location, A_visual, A_food_here, A_viewpoint],
        B=[B_location, B_food],
        C=[C_location, C_visual, C_food_here, C_viewpoint],
        D=[np.asarray(D_location, dtype=float), np.asarray(D_food, dtype=float)],
        A_dependencies=[[0], [0, 1], [0, 1], [0]],
        B_dependencies=[[0], [1]],
        control_fac_idx=[0],
    )

    agent = make_agent_instance(Agent,
        A=pymdp_model.A,
        B=pymdp_model.B,
        C=pymdp_model.C,
        D=pymdp_model.D,
        E=None,
        A_dependencies=pymdp_model.A_dependencies,
        B_dependencies=pymdp_model.B_dependencies,
        control_fac_idx=pymdp_model.control_fac_idx,
        policy_len=policy_len,
        gamma=model.gamma,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        action_selection="stochastic",
        sampling_mode="full",
        inference_algo="fpi",
        batch_size=1,
    )
    return agent, pymdp_model


def make_agent_with_policy_prior(
    world: GridWorld,
    model: ModelSpec,
    location_prior: np.ndarray,
    food_prior: np.ndarray,
    policy_prior: np.ndarray | None,
    policy_len: int,
) -> Any:
    _, _, Agent = require_pymdp()
    tmp_agent, pymdp_model = make_pymdp_model(world, model, location_prior, food_prior, policy_len=policy_len)
    policies = get_policy_array(tmp_agent)
    if policy_prior is None:
        policy_prior = np.ones(policies.shape[0], dtype=float) / policies.shape[0]
    return make_agent_instance(Agent,
        A=pymdp_model.A,
        B=pymdp_model.B,
        C=pymdp_model.C,
        D=pymdp_model.D,
        E=np.asarray(policy_prior, dtype=float),
        A_dependencies=pymdp_model.A_dependencies,
        B_dependencies=pymdp_model.B_dependencies,
        control_fac_idx=pymdp_model.control_fac_idx,
        policy_len=policy_len,
        gamma=model.gamma,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        action_selection="stochastic",
        sampling_mode="full",
        inference_algo="fpi",
        batch_size=1,
    )


def get_policy_array(agent: Any) -> np.ndarray:
    arr = np.asarray(agent.policies.policy_arr)
    if arr.ndim == 2:
        # Legacy/simple form: (n_policies, policy_len) for one control factor.
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected pymdp policy array shape: {arr.shape}")
    return arr.astype(int)


def get_factor_belief(qs_factor: Any, n_states: int) -> np.ndarray:
    arr = np.asarray(qs_factor, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        # Batched JAX pymdp commonly returns (batch, states). Older/legacy
        # forms may return (time, states). In this experiment batch_size=1,
        # so the last row is safe for either convention.
        arr = arr[-1]
    arr = arr.reshape(-1)
    if arr.size != n_states:
        raise RuntimeError(f"Unexpected posterior factor shape {np.asarray(qs_factor).shape}; expected {n_states} states.")
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones(n_states) / n_states
    return arr / s


def batch_observations(obs: list[int]) -> list[np.ndarray]:
    """Return observations with an explicit batch axis for JAX pymdp.

    Current pymdp's JAX Agent vectorises inference over batch axis 0. Passing
    bare Python integers causes pymdp to expand observations into unbatched
    one-hot vectors of shape (n_observations,), which then makes vmap confuse
    observation categories with batch items. A one-element NumPy integer array
    gives the intended batch size of one without passing caller-side JAX arrays
    into Equinox static fields.
    """
    return [np.asarray([int(o)], dtype=np.int32) for o in obs]


def one_hot(n: int, idx: int) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    x[int(idx)] = 1.0
    return x


def observe_world(world: GridWorld, loc: int, food_cell: int, rng: np.random.Generator, cfg: Config) -> list[int]:
    location_obs = int(loc)
    if world.los_visible[loc, food_cell] and rng.random() < cfg.los_detection_prob:
        visual_obs = int(food_cell)
    else:
        visual_obs = world.n_cells
        if rng.random() < cfg.los_false_positive_prob:
            visible = np.where(world.los_visible[loc])[0]
            if visible.size:
                visual_obs = int(rng.choice(visible))
    if loc == food_cell:
        food_here_obs = int(rng.random() < cfg.food_here_detection_prob)
    else:
        food_here_obs = int(rng.random() < 0.02)
    viewpoint_obs = int(world.viewpoints[loc])
    return [location_obs, visual_obs, food_here_obs, viewpoint_obs]


def expected_visual_information_gain(world: GridWorld, loc: int, q_food: np.ndarray, cfg: Config) -> float:
    n = world.n_cells
    q_food = np.asarray(q_food, dtype=float)
    q_food = q_food / max(q_food.sum(), 1e-12)
    prior_entropy = entropy(q_food)
    A_visual = np.zeros((n + 1, n), dtype=float)
    visible = world.los_visible[loc]
    for food in range(n):
        if visible[food]:
            A_visual[food, food] = cfg.los_detection_prob
            A_visual[n, food] = 1.0 - cfg.los_detection_prob
        else:
            A_visual[n, food] = 1.0 - cfg.los_false_positive_prob
            visible_idx = np.where(visible)[0]
            if visible_idx.size:
                A_visual[visible_idx, food] += cfg.los_false_positive_prob / visible_idx.size
            else:
                A_visual[n, food] = 1.0
    p_obs = A_visual @ q_food
    expected_post_entropy = 0.0
    for obs in np.where(p_obs > 1e-12)[0]:
        likelihood = A_visual[obs]
        post = likelihood * q_food
        post = post / max(post.sum(), 1e-12)
        expected_post_entropy += float(p_obs[obs]) * entropy(post)
    return max(0.0, prior_entropy - expected_post_entropy)


def rollout_policy_features(world: GridWorld, start_loc: int, policy: np.ndarray, q_food: np.ndarray, cfg: Config) -> dict[str, float]:
    loc = int(start_loc)
    cum_cost = 0.0
    cum_scan_cost = 0.0
    prospects: list[float] = []
    refuges: list[float] = []
    stationarities: list[float] = []
    attainments: list[float] = []
    resources: list[float] = []
    hazards: list[float] = []
    info_gains: list[float] = []
    first_action = int(policy[0, 0])
    illegal_first_action = not bool(world.legal_action[loc, first_action])

    for t in range(policy.shape[0]):
        action = int(policy[t, 0])
        cum_cost += float(world.movement_cost[loc, action])
        if ACTION_NAMES[action] == "scan":
            cum_scan_cost += cfg.scan_cost
        loc = int(world.transition_to[loc, action])
        prospects.append(float(world.prospect[loc]))
        refuges.append(float(world.refuge[loc]))
        stationarities.append(float(world.stationarity[loc]))
        attainments.append(float(world.attainment[loc]))
        resources.append(float(world.resource[loc]))
        hazards.append(float(world.hazard[loc]))
        if ACTION_NAMES[action] == "scan" or world.viewpoints[loc]:
            info_gains.append(expected_visual_information_gain(world, loc, q_food, cfg))

    mean_cost = cum_cost + cum_scan_cost + 1e-4
    info = float(np.sum(info_gains)) if info_gains else expected_visual_information_gain(world, loc, q_food, cfg) * 0.25
    return {
        "movement_cost": mean_cost,
        "prospect": float(np.mean(prospects)),
        "refuge": float(np.mean(refuges)),
        "stationarity": float(np.mean(stationarities)),
        "attainment": float(np.mean(attainments)),
        "resource": float(np.mean(resources)),
        "hazard": float(np.mean(hazards)),
        "efficient_information": info / mean_cost,
        "terminal_information": info,
        "illegal_first_action": float(illegal_first_action),
    }


def compute_policy_prior(world: GridWorld, model: ModelSpec, policies: np.ndarray, loc: int, q_food: np.ndarray, cfg: Config) -> np.ndarray:
    scores = np.zeros(policies.shape[0], dtype=float)
    for p_idx, policy in enumerate(policies):
        feats = rollout_policy_features(world, loc, policy, q_food, cfg)
        score = 0.0
        for name, weight in model.weights.items():
            score += float(weight) * float(feats.get(name, 0.0))
        score -= 8.0 * feats["illegal_first_action"]
        scores[p_idx] = score
    # Standardise within the current policy set. This prevents one feature scale
    # from trivially dominating model comparison.
    if np.nanstd(scores) > 1e-9:
        scores = (scores - np.nanmean(scores)) / np.nanstd(scores)
    prior = softmax(scores, temperature=model.policy_prior_temperature)
    prior = np.maximum(prior, cfg.min_policy_prior)
    return prior / prior.sum()


def action_probs_from_policies(q_pi: Any, policies: np.ndarray) -> np.ndarray:
    q = np.asarray(q_pi, dtype=float)
    q = np.squeeze(q)
    if q.ndim != 1:
        q = q.reshape(-1)
    probs = np.zeros(N_ACTIONS, dtype=float)
    for p_idx, policy in enumerate(policies):
        action = int(policy[0, 0])
        if 0 <= action < N_ACTIONS:
            probs[action] += q[p_idx]
    if probs.sum() <= 0 or not np.isfinite(probs.sum()):
        probs[:] = 1.0 / N_ACTIONS
    else:
        probs /= probs.sum()
    return probs


def choose_action(action_probs: np.ndarray, rng: np.random.Generator) -> int:
    p = np.asarray(action_probs, dtype=float)
    p = p / max(p.sum(), 1e-12)
    return int(rng.choice(np.arange(N_ACTIONS), p=p))


def infer_action_distribution(
    world: GridWorld,
    model: ModelSpec,
    loc: int,
    obs: list[int],
    prior_location: np.ndarray,
    prior_food: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray, np.ndarray, np.ndarray]:
    # Build a temporary agent only to get posterior beliefs and policy layout.
    tmp_agent = make_agent_with_policy_prior(
        world,
        model,
        prior_location,
        prior_food,
        policy_prior=None,
        policy_len=cfg.policy_len,
    )
    qs = tmp_agent.infer_states(batch_observations(obs), empirical_prior=tmp_agent.D)
    q_loc = get_factor_belief(qs[0], world.n_cells)
    q_food = get_factor_belief(qs[1], world.n_cells)
    policies = get_policy_array(tmp_agent)
    E = compute_policy_prior(world, model, policies, loc, q_food, cfg)
    agent = make_agent_with_policy_prior(world, model, prior_location, prior_food, E, cfg.policy_len)
    q_pi, neg_efe = agent.infer_policies(qs)
    action_probs = action_probs_from_policies(q_pi, policies)
    return action_probs, q_food, qs, policies, np.asarray(q_pi), np.asarray(neg_efe)


def simulate_trial(
    world: GridWorld,
    model: ModelSpec,
    cfg: Config,
    rng: np.random.Generator,
    agent_id: int,
    trial_id: int,
    true_model_name: str | None = None,
) -> TrialSimulation:
    true_model_name = true_model_name or model.name
    start_loc = int(rng.choice(world.start_cells))
    food_cell = int(rng.choice(np.arange(world.n_cells), p=world.food_prior))
    loc = start_loc
    energy = cfg.start_energy
    eaten = False
    visited = np.zeros(world.n_cells, dtype=int)
    visited[loc] += 1
    scan_count = 0
    dwell_count = 0
    entropy_reduction = 0.0
    resource_gain = 0.0
    hazard_penalty = 0.0
    viewpoint_visits = 0
    decisions: list[dict[str, Any]] = []

    prior_location = one_hot(world.n_cells, loc)
    prior_food = world.food_prior.copy()
    previous_food_entropy = entropy(prior_food)

    for t in range(cfg.horizon):
        obs = observe_world(world, loc, food_cell, rng, cfg)
        action_probs, q_food, qs, policies, q_pi, neg_efe = infer_action_distribution(
            world, model, loc, obs, prior_location, prior_food, cfg
        )
        action_idx = choose_action(action_probs, rng)
        action_name = ACTION_NAMES[action_idx]
        chosen_prob = float(max(action_probs[action_idx], 1e-12))

        current_food_entropy = entropy(q_food)
        entropy_step = max(0.0, previous_food_entropy - current_food_entropy)
        entropy_reduction += entropy_step
        previous_food_entropy = current_food_entropy

        old_loc = loc
        next_loc = int(world.transition_to[loc, action_idx])
        step_cost = float(world.movement_cost[loc, action_idx])
        if action_name == "scan":
            scan_count += 1
            energy -= cfg.scan_cost * (1.0 + 0.35 * (1.0 - world.legibility[loc]))
        elif action_name == "stay":
            dwell_count += 1
            energy -= cfg.stay_cost * (1.0 - world.stationarity[loc])
        else:
            energy -= cfg.base_metabolic_cost + cfg.move_cost_scale * step_cost
        loc = next_loc
        visited[loc] += 1
        viewpoint_visits += int(world.viewpoints[loc])

        hcost = cfg.hazard_cost_scale * float(world.hazard[loc])
        hazard_penalty += hcost
        energy -= hcost
        if loc == food_cell:
            eaten = True
            gain = cfg.food_gain * float(0.5 + 0.5 * world.resource[loc])
            resource_gain += gain
            energy += gain
        energy = float(np.clip(energy, 0.0, 1.75))

        decisions.append(
            {
                "world_id": world.id,
                "family": world.family,
                "level": world.level,
                "generator_model": true_model_name,
                "agent_id": agent_id,
                "trial": trial_id,
                "t": t,
                "location": int(old_loc),
                "next_location": int(loc),
                "food_cell": int(food_cell),
                "obs_location": int(obs[0]),
                "obs_visual": int(obs[1]),
                "obs_food_here": int(obs[2]),
                "obs_viewpoint": int(obs[3]),
                "chosen_action": action_name,
                "chosen_action_idx": int(action_idx),
                "chosen_probability": chosen_prob,
                "energy_before_action": float(energy),
                "posterior_food_entropy": float(current_food_entropy),
                "entropy_reduction_step": float(entropy_step),
                "prospect": float(world.prospect[old_loc]),
                "refuge": float(world.refuge[old_loc]),
                "stationarity": float(world.stationarity[old_loc]),
                "attainment": float(world.attainment[old_loc]),
                "resource": float(world.resource[old_loc]),
                "hazard": float(world.hazard[old_loc]),
                "movement_cost": float(step_cost),
                "is_viewpoint": int(world.viewpoints[old_loc]),
                "q_pi_max": float(np.max(q_pi)),
                "neg_efe_best": float(np.max(neg_efe)),
            }
        )

        # For the next timestep, location is fully observed and food belief carries over.
        prior_location = one_hot(world.n_cells, loc)
        prior_food = q_food.copy()
        if eaten:
            # Keep the trial length comparable but reset target if food is eaten;
            # this produces resource-seeking behaviour rather than a trivial stop.
            food_cell = int(rng.choice(np.arange(world.n_cells), p=world.food_prior))
            prior_food = world.food_prior.copy()
            previous_food_entropy = entropy(prior_food)
            eaten = False

    unique_viewpoints = int(np.sum((visited > 0) & world.viewpoints))
    total_viewpoint_visits = int(np.sum(visited[world.viewpoints]))
    revisits = max(0, total_viewpoint_visits - unique_viewpoints)
    ended_at_viewpoint = int(world.viewpoints[loc])
    view_choice = int(total_viewpoint_visits > 0)
    regulation_success = int(energy > cfg.energy_min_success and entropy_reduction > 0.25 and hazard_penalty < 0.75)

    trial_row = {
        "world_id": world.id,
        "family": world.family,
        "level": world.level,
        "generator_model": true_model_name,
        "agent_id": agent_id,
        "trial": trial_id,
        "start_location": int(start_loc),
        "final_location": int(loc),
        "viewpoint_choice_probability": float(view_choice),
        "ended_at_viewpoint": float(ended_at_viewpoint),
        "scan_choice_probability": float(scan_count / cfg.horizon),
        "dwell_time_at_viewpoint": float(sum(1 for d in decisions if d["chosen_action"] in {"stay", "scan"} and d["is_viewpoint"] == 1) / cfg.horizon),
        "viewpoint_occupancy": float(total_viewpoint_visits / cfg.horizon),
        "revisit_rate": float(revisits / max(1, total_viewpoint_visits)),
        "entropy_reduction": float(entropy_reduction),
        "final_energy": float(energy),
        "hazard_penalty": float(hazard_penalty),
        "resource_gain": float(resource_gain),
        "regulation_success": float(regulation_success),
    }
    return TrialSimulation(trial_row=trial_row, decision_rows=decisions)


def world_summary_rows(worlds: Iterable[GridWorld]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for w in worlds:
        rows.append(
            {
                "world_id": w.id,
                "family": w.family,
                "level": w.level,
                "level_numeric": w.level_numeric,
                "n_cells": w.n_cells,
                "n_viewpoints": int(np.sum(w.viewpoints)),
                "mean_height": float(np.mean(w.height)),
                "mean_slope": float(np.mean(w.slope)),
                "mean_prospect": float(np.mean(w.prospect)),
                "viewpoint_prospect": float(np.mean(w.prospect[w.viewpoints])),
                "viewpoint_refuge": float(np.mean(w.refuge[w.viewpoints])),
                "viewpoint_stationarity": float(np.mean(w.stationarity[w.viewpoints])),
                "viewpoint_attainment": float(np.mean(w.attainment[w.viewpoints])),
                "mean_resource": float(np.mean(w.resource)),
                "mean_hazard": float(np.mean(w.hazard)),
            }
        )
    return rows


def replay_log_likelihood(
    decision_rows: pd.DataFrame,
    world_map: dict[str, GridWorld],
    candidate_model: ModelSpec,
    cfg: Config,
) -> float:
    ll = 0.0
    grouped = decision_rows.sort_values(["world_id", "agent_id", "trial", "t"]).groupby(["world_id", "agent_id", "trial"], sort=False)
    for (world_id, _agent_id, _trial), D in grouped:
        world = world_map[str(world_id)]
        first = D.iloc[0]
        loc = int(first["location"])
        prior_location = one_hot(world.n_cells, loc)
        prior_food = world.food_prior.copy()
        for _, row in D.iterrows():
            loc = int(row["location"])
            obs = [int(row["obs_location"]), int(row["obs_visual"]), int(row["obs_food_here"]), int(row["obs_viewpoint"])]
            action_probs, q_food, *_ = infer_action_distribution(world, candidate_model, loc, obs, prior_location, prior_food, cfg)
            chosen = int(row["chosen_action_idx"])
            ll += math.log(max(float(action_probs[chosen]), 1e-12))
            prior_location = one_hot(world.n_cells, int(row["next_location"]))
            prior_food = q_food.copy()
    return float(ll)


def run_model_recovery(decision_df: pd.DataFrame, worlds: list[GridWorld], models: list[ModelSpec], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    world_map = {w.id: w for w in worlds}
    true_models = sorted(decision_df["generator_model"].unique().tolist())
    rows = []
    total_fits = len(true_models) * len(models)
    with progress_bar(total_fits, "Model recovery", "fit", cfg) as pbar:
        for true_name in true_models:
            D = decision_df[decision_df["generator_model"] == true_name].copy()
            for candidate in models:
                pbar.set_postfix_str(f"true={true_name}, fit={candidate.name}")
                ll = replay_log_likelihood(D, world_map, candidate, cfg)
                rows.append(
                    {
                        "true_model": true_name,
                        "candidate_model": candidate.name,
                        "n_decisions": int(len(D)),
                        "log_likelihood": ll,
                        "mean_log_likelihood": ll / max(1, len(D)),
                    }
                )
                pbar.update(1)
    loglik = pd.DataFrame(rows)
    best_rows = []
    for true_name, sub in loglik.groupby("true_model"):
        sub = sub.sort_values("log_likelihood", ascending=False).reset_index(drop=True)
        margin = float(sub.loc[0, "log_likelihood"] - sub.loc[1, "log_likelihood"]) if len(sub) > 1 else float("nan")
        best_rows.append(
            {
                "true_model": true_name,
                "best_recovered_model": sub.loc[0, "candidate_model"],
                "loglik_margin": margin,
                "n_decisions": int(sub.loc[0, "n_decisions"]),
                "recovered_correctly": int(sub.loc[0, "candidate_model"] == true_name),
            }
        )
    ident = pd.DataFrame(best_rows)
    confusion = pd.crosstab(ident["true_model"], ident["best_recovered_model"])
    for m in [m.name for m in models]:
        if m not in confusion.columns:
            confusion[m] = 0
    confusion = confusion[[m.name for m in models]].reset_index()
    return loglik, ident, confusion


def primary_analysis(trial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_vars = [
        "viewpoint_choice_probability",
        "scan_choice_probability",
        "dwell_time_at_viewpoint",
        "revisit_rate",
        "entropy_reduction",
        "regulation_success",
    ]
    summary = (
        trial_df.groupby(["family", "level", "generator_model"], dropna=False)[primary_vars]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in summary.columns]

    contrast_rows = []
    for dv in primary_vars:
        for (family, generator), sub in trial_df.groupby(["family", "generator_model"]):
            means = sub.groupby("level")[dv].mean()
            if "low" in means and "high" in means:
                contrast_rows.append(
                    {
                        "dependent_variable": dv,
                        "family": family,
                        "generator_model": generator,
                        "low_mean": float(means["low"]),
                        "high_mean": float(means["high"]),
                        "high_minus_low": float(means["high"] - means["low"]),
                    }
                )
    contrasts = pd.DataFrame(contrast_rows)
    return summary, contrasts


def posterior_predictive_checks(
    trial_df: pd.DataFrame,
    ident_df: pd.DataFrame,
    worlds: list[GridWorld],
    models: list[ModelSpec],
    cfg: Config,
) -> pd.DataFrame:
    model_map = {m.name: m for m in models}
    rng = np.random.default_rng(cfg.seed + 9090)
    rows = []
    ppc_worlds = worlds[: max(1, min(len(worlds), 3))]
    outcomes = [
        "viewpoint_choice_probability",
        "scan_choice_probability",
        "dwell_time_at_viewpoint",
        "revisit_rate",
        "entropy_reduction",
        "regulation_success",
    ]
    total_ppc_trials = len(ident_df) * cfg.ppc_replications * len(ppc_worlds)
    with progress_bar(total_ppc_trials, "Posterior predictive checks", "trial", cfg) as pbar:
        for _, rec in ident_df.iterrows():
            true_name = str(rec["true_model"])
            recovered_name = str(rec["best_recovered_model"])
            model = model_map[recovered_name]
            obs = trial_df[trial_df["generator_model"] == true_name][outcomes].mean(numeric_only=True)
            sim_rows = []
            for rep in range(cfg.ppc_replications):
                for w in ppc_worlds:
                    pbar.set_postfix_str(f"model={recovered_name}")
                    sim = simulate_trial(w, model, cfg, rng, agent_id=10_000 + rep, trial_id=rep, true_model_name=recovered_name)
                    sim_rows.append(sim.trial_row)
                    pbar.update(1)
            rep_mean = pd.DataFrame(sim_rows)[outcomes].mean(numeric_only=True)
            row = {"true_model": true_name, "ppc_model": recovered_name}
            for dv in outcomes:
                row[f"observed_{dv}"] = float(obs[dv])
                row[f"replicated_{dv}"] = float(rep_mean[dv])
                row[f"delta_{dv}"] = float(rep_mean[dv] - obs[dv])
            rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(cfg: Config) -> dict[str, Any]:
    ensure_dirs(cfg)
    started = time.perf_counter()
    print(f"Started pymdp Study 1 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    models = specify_candidate_models()
    worlds = generate_world_bank(cfg)

    total_trials = len(worlds) * len(models) * cfg.n_agents * cfg.n_trials_per_agent
    estimated_decisions = total_trials * cfg.horizon
    recovery_fits = len(models) * len(models)
    print(
        "Planned work: "
        f"{len(worlds)} worlds, {len(models)} models, {total_trials} trials, "
        f"~{estimated_decisions} decisions, {recovery_fits} model-recovery fits. "
        "Progress bars report elapsed time and ETA."
    )

    trial_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    with progress_bar(total_trials, "Simulating trials", "trial", cfg) as pbar:
        for world in worlds:
            for model in models:
                for agent_id in range(cfg.n_agents):
                    for trial in range(cfg.n_trials_per_agent):
                        pbar.set_postfix_str(f"world={world.id}, model={model.name}")
                        sim = simulate_trial(world, model, cfg, rng, agent_id=agent_id, trial_id=trial, true_model_name=model.name)
                        trial_rows.append(sim.trial_row)
                        decision_rows.extend(sim.decision_rows)
                        pbar.update(1)

    trial_df = pd.DataFrame(trial_rows)
    decision_df = pd.DataFrame(decision_rows)
    world_df = pd.DataFrame(world_summary_rows(worlds))
    model_df = pd.DataFrame([
        {"model": m.name, "weights": json.dumps(m.weights), "description": m.description, "gamma": m.gamma}
        for m in models
    ])
    family_df = pd.DataFrame([asdict(f) for f in define_world_families()])

    primary_summary, primary_contrasts = primary_analysis(trial_df)
    loglik, ident, confusion = run_model_recovery(decision_df, worlds, models, cfg)
    ppc = posterior_predictive_checks(trial_df, ident, worlds, models, cfg)

    outputs = {
        "trials": trial_df,
        "decisions": decision_df,
        "worlds": world_df,
        "models": model_df,
        "families": family_df,
        "primary_summary": primary_summary,
        "primary_contrasts": primary_contrasts,
        "model_recovery_loglik": loglik,
        "model_recovery_identifiability": ident,
        "model_recovery_confusion": confusion,
        "posterior_predictive_checks": ppc,
        "world_objects": worlds,
    }
    write_outputs(outputs, cfg)
    elapsed = time.perf_counter() - started
    print(f"Finished pymdp Study 1 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} after {format_duration(elapsed)}.")
    return outputs


def write_outputs(outputs: dict[str, Any], cfg: Config) -> None:
    for key, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(cfg.tables_dir / f"{key}.csv", index=False)
    manifest = {
        "project": "viewpoint_active_inference_pymdp_study1",
        "seed": cfg.seed,
        "test_mode": cfg.test_mode,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "candidate_models": outputs["models"].to_dict(orient="records"),
        "outputs": {k: str(cfg.tables_dir / f"{k}.csv") for k, v in outputs.items() if isinstance(v, pd.DataFrame)},
    }
    (cfg.json_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(outputs, cfg)
    plot_overview(outputs, cfg)
    plot_example_world(outputs["world_objects"][0], cfg)


def write_summary(outputs: dict[str, Any], cfg: Config) -> None:
    ident = outputs["model_recovery_identifiability"]
    recovery_accuracy = float(ident["recovered_correctly"].mean()) if not ident.empty else float("nan")
    primary = outputs["primary_summary"]
    lines = [
        "Study 1 pymdp active-inference viewpoint model discovery",
        f"seed={cfg.seed}",
        f"test_mode={cfg.test_mode}",
        f"n_trials={len(outputs['trials'])}",
        f"n_decisions={len(outputs['decisions'])}",
        f"n_worlds={len(outputs['worlds'])}",
        f"model_recovery_accuracy={recovery_accuracy:.3f}",
        "",
        "Recovered model by generator:",
    ]
    for _, row in ident.iterrows():
        lines.append(
            f"  {row['true_model']} -> {row['best_recovered_model']} "
            f"(margin={row['loglik_margin']:.3f}, n={int(row['n_decisions'])})"
        )
    lines.extend([
        "",
        "Primary behavioural table saved to tables/primary_summary.csv.",
        "Model recovery likelihoods saved to tables/model_recovery_loglik.csv.",
        "Posterior predictive checks saved to tables/posterior_predictive_checks.csv.",
        "",
        "Interpretation guardrail: this is a model-discovery simulation. It does not establish that real-world viewpoints are selected by the winning mechanism; that is the role of Study 2.",
    ])
    (cfg.output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_overview(outputs: dict[str, Any], cfg: Config) -> None:
    trials = outputs["trials"].copy()
    ident = outputs["model_recovery_identifiability"].copy()
    confusion = outputs["model_recovery_confusion"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    g = trials.groupby("generator_model", sort=False)["viewpoint_choice_probability"].mean().sort_values(ascending=False)
    axes[0, 0].bar(g.index.astype(str), g.values)
    axes[0, 0].set_ylabel("Mean viewpoint choice")
    axes[0, 0].set_title("Viewpoint choice by generator")
    axes[0, 0].tick_params(axis="x", rotation=35)

    g2 = trials.groupby("generator_model", sort=False)["entropy_reduction"].mean().reindex(g.index)
    axes[0, 1].bar(g2.index.astype(str), g2.values)
    axes[0, 1].set_ylabel("Mean entropy reduction")
    axes[0, 1].set_title("Information gain by generator")
    axes[0, 1].tick_params(axis="x", rotation=35)

    mat = confusion.drop(columns=["true_model"]).to_numpy(dtype=float)
    im = axes[1, 0].imshow(mat, aspect="auto")
    axes[1, 0].set_xticks(range(mat.shape[1]))
    axes[1, 0].set_xticklabels(confusion.columns[1:], rotation=45, ha="right")
    axes[1, 0].set_yticks(range(mat.shape[0]))
    axes[1, 0].set_yticklabels(confusion["true_model"].astype(str).tolist())
    axes[1, 0].set_title("Model recovery confusion")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].scatter(trials["entropy_reduction"], trials["final_energy"], s=10, alpha=0.25)
    axes[1, 1].set_xlabel("Entropy reduction")
    axes[1, 1].set_ylabel("Final energy")
    axes[1, 1].set_title("Raw behaviour: information and regulation")

    fig.suptitle("pymdp Study 1 active-inference viewpoint simulation", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, cfg.figures_dir / "study1_pymdp_overview")


def plot_example_world(world: GridWorld, cfg: Config) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        (world.height, "Elevation"),
        (world.prospect.reshape(world.size, world.size), "Prospect / line of sight"),
        (world.stationarity.reshape(world.size, world.size), "Stationarity"),
        (world.refuge.reshape(world.size, world.size), "Refuge"),
        (world.resource.reshape(world.size, world.size), "Food prior/resource"),
        (world.hazard.reshape(world.size, world.size), "Hazard"),
    ]
    for ax, (data, title) in zip(axes.flat, panels):
        im = ax.imshow(data, origin="upper")
        vp_r, vp_c = np.where(world.viewpoints.reshape(world.size, world.size))
        ax.scatter(vp_c, vp_r, s=18, marker="o", facecolors="none", edgecolors="white", linewidths=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Example synthetic topographic world: {world.id}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, cfg.figures_dir / "example_world_layers")


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = make_config(args)
    try:
        run_experiment(cfg)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Completed pymdp Study 1 simulation. Outputs written to: {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())