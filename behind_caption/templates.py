import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def list_templates() -> list[str]:
    return sorted(p.stem for p in TEMPLATE_DIR.glob("*.json"))


def get_template(name: str) -> dict:
    p = TEMPLATE_DIR / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"Unknown template '{name}'. Available: {', '.join(list_templates())}")
    return json.loads(p.read_text())
