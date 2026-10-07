"""Experiment configuration.

The YAML file defines one baseline and a list of experiments. Each experiment
varies exactly one field of the baseline, so every run differs from the
baseline in a single variable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    name: str
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding: str = "gemini-embedding-001"
    prompt: str = "v1_basic"
    top_k: int = 4
    generator_model: str = "gemini-2.5-flash-lite"
    judge_model: str = "gemini-3.1-flash-lite"
    temperature: float | None = None

    def index_key(self) -> str:
        return f"{self.embedding}__cs{self.chunk_size}__ov{self.chunk_overlap}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShipGates:
    """Pre-registered thresholds a config must meet to be recommended."""

    faithfulness_pass_rate: float = 0.90
    relevance_pass_rate: float = 0.85
    unanswerable_safe_rate: float = 0.80
    false_abstention_rate: float = 0.10
    retrieval_recall_at_k: float = 0.85


@dataclass
class ExperimentPlan:
    baseline: RunConfig
    experiments: dict[str, list[RunConfig]]
    gates: ShipGates = field(default_factory=ShipGates)
    rpm: int = 30
    prices_per_million: dict[str, dict[str, float]] = field(default_factory=dict)

    def all_runs(self) -> list[RunConfig]:
        seen: dict[str, RunConfig] = {self.baseline.name: self.baseline}
        for runs in self.experiments.values():
            for run in runs:
                seen.setdefault(run.name, run)
        return list(seen.values())


def load_plan(path: str | Path) -> ExperimentPlan:
    raw = yaml.safe_load(Path(path).read_text())
    base_fields = dict(raw["baseline"])
    baseline = RunConfig(name="baseline", **base_fields)

    experiments: dict[str, list[RunConfig]] = {}
    for exp in raw.get("experiments", []):
        variable = exp["variable"]
        if variable not in RunConfig.__dataclass_fields__ or variable == "name":
            raise ValueError(f"Unknown experiment variable: {variable}")
        runs = []
        for value in exp["values"]:
            if value == getattr(baseline, variable):
                runs.append(baseline)
            else:
                runs.append(replace(baseline, name=f"{variable}={value}", **{variable: value}))
        experiments[exp["name"]] = runs

    gates = ShipGates(**raw.get("ship_gates", {}))
    return ExperimentPlan(
        baseline=baseline,
        experiments=experiments,
        gates=gates,
        rpm=int(raw.get("requests_per_minute", 30)),
        prices_per_million=raw.get("prices_per_million", {}) or {},
    )
