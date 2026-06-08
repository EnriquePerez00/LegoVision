# -*- coding: utf-8 -*-
"""
LegoVision — Config Loader
===========================
Carga `config.yaml` y expone todos los parámetros como atributos anidados.

Uso:
    from config_loader import cfg

    # Acceso con punto (dot notation):
    cfg.scene.belt.width_bu           # → 20.0
    cfg.cameras.cenital.position      # → [0.0, 0.0, 15.0]
    cfg.yolo.training.epochs          # → 35
    cfg.inference.segmentation.method # → "chromaticity"

    # Acceso como dict:
    cfg["scene"]["belt"]["width_bu"]  # → 20.0

    # Override puntual (no persiste al fichero):
    cfg.inference.segmentation.method = "otsu"
"""
import os
import yaml


class _CfgNode:
    """Nodo de configuración con acceso por atributo y por clave."""

    def __init__(self, data: dict):
        for key, value in data.items():
            attr_name = str(key)
            if isinstance(value, dict):
                # Si TODAS las claves son strings → nodo anidado; si no → dict plano
                if all(isinstance(k, str) for k in value.keys()):
                    setattr(self, attr_name, _CfgNode(value))
                else:
                    setattr(self, attr_name, value)
            else:
                setattr(self, attr_name, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        items = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"CfgNode({items})"

    def to_dict(self) -> dict:
        """Convierte recursivamente a dict nativo."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, _CfgNode):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def get(self, key, default=None):
        return getattr(self, key, default)


def _find_config() -> str:
    """Busca config.yaml subiendo desde el directorio actual hasta la raíz del proyecto."""
    # 1. Intentar junto a este fichero
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "config.yaml")
    if os.path.exists(candidate):
        return candidate

    # 2. Subir directorios buscando config.yaml (máx 5 niveles)
    d = here
    for _ in range(5):
        d = os.path.dirname(d)
        candidate = os.path.join(d, "config.yaml")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "config.yaml no encontrado. "
        "Asegúrate de que existe en la raíz del proyecto LegoVision."
    )


def load_config(path: str = None) -> _CfgNode:
    """Carga el YAML y devuelve un árbol de CfgNode."""
    if path is None:
        path = _find_config()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _CfgNode(raw)


# Singleton: se carga una sola vez al importar el módulo
cfg = load_config()
