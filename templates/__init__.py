import os
import json
from .base import BaseTemplate
from .dynamic_template import DynamicTemplate
from .luxe_template import LuxeTemplate

def load_template_config(name: str) -> dict:
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    config_path = os.path.join(config_dir, f"{name.lower()}.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_template(name: str) -> BaseTemplate:
    """
    Returns an initialized template instance based on style name:
    - 'dynamic' (Estate Dynamic bold look)
    - 'luxe' (Luxe premium elegant look)
    """
    name_lower = name.lower()
    config = load_template_config(name_lower)
    
    if name_lower in ["dynamic", "estate_dynamic", "estate dynamic"]:
        return DynamicTemplate(config)
    elif name_lower in ["luxe", "luxury", "elegant"]:
        return LuxeTemplate(config)
    else:
        raise ValueError(f"Unknown template style: '{name}'. Available: ['dynamic', 'luxe']")

__all__ = ["BaseTemplate", "DynamicTemplate", "LuxeTemplate", "get_template"]
