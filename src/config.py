"""Load and validate config.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "pulse.db"


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights = cfg.get("health_weights", {})
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"health_weights must sum to 1.0 (got {total})")

    return cfg


def entities(cfg: dict) -> list[dict]:
    """Brand first, then competitors — this fixed order drives chart colors."""
    return [cfg["brand"], *cfg.get("competitors", [])]
