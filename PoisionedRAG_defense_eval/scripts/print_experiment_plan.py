from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defense.registry import DATASET_SPECS, DEFENSE_SPECS, MODEL_SPECS


def main() -> None:
    combos = len(MODEL_SPECS) * len(DATASET_SPECS) * len(DEFENSE_SPECS)
    print("Defense study plan")
    print(f"Models: {len(MODEL_SPECS)}")
    print(f"Datasets: {len(DATASET_SPECS)}")
    print(f"Defenses: {len(DEFENSE_SPECS)}")
    print(f"Matrix size: {combos}")
    print("")
    print("Defenses:")
    for item in DEFENSE_SPECS:
        print(f"- {item.key}: {item.description}")
    print("")
    print("Models:")
    for item in MODEL_SPECS:
        print(f"- {item.key}: {item.model_name} [{item.provider}]")
    print("")
    print("Datasets:")
    for item in DATASET_SPECS:
        print(f"- {item.key}: {item.slug}")


if __name__ == "__main__":
    main()
