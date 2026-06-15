# -*- coding: utf-8 -*-
"""
LegoVision — Logger Centralizado
==================================
Módulo de logging configurable desde config.yaml.

Uso en cualquier script:
    from logger import get_logger
    log = get_logger("blender")          # → logs/blender_YYYY-MM-DD.log
    log = get_logger("dinov2")           # → logs/dinov2_YYYY-MM-DD.log
    log = get_logger("pipeline")         # → logs/pipeline_YYYY-MM-DD.log
    log = get_logger("mi_script")        # → logs/mi_script_YYYY-MM-DD.log

Los logs se escriben en:
    2camaras_multi_pieza/logs/<modulo>_<YYYY-MM-DD>.log

Y simultáneamente en stdout para visibilidad en consola/Blender.

Configuración en config.yaml (sección 12):
    logging:
      enabled: true
      dir: "logs"
      level: "INFO"
      max_file_size_mb: 50
      max_files: 10
"""
import os
import sys
import logging
import logging.handlers
from datetime import date

# ── Resolución de paths ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))


def _get_log_dir() -> str:
    """Obtiene el directorio de logs desde config.yaml o usa el default."""
    try:
        from config_loader import cfg
        log_dir = getattr(cfg, "logging", None)
        if log_dir:
            d = getattr(log_dir, "dir", "logs")
        else:
            d = "logs"
    except Exception:
        d = "logs"
    # Siempre relativo a la raíz del proyecto (2camaras_multi_pieza/)
    if not os.path.isabs(d):
        d = os.path.join(_HERE, d)
    os.makedirs(d, exist_ok=True)
    return d


def _get_config(section: str) -> dict:
    """Lee la configuración de logging del config.yaml."""
    defaults = {
        "enabled": True,
        "level": "INFO",
        "max_file_size_mb": 50,
        "max_files": 10,
    }
    try:
        from config_loader import cfg
        log_cfg = getattr(cfg, "logging", None)
        if not log_cfg:
            return defaults
        # Override con config global
        defaults["enabled"] = getattr(log_cfg, "enabled", True)
        defaults["level"] = getattr(log_cfg, "level", "INFO")
        defaults["max_file_size_mb"] = getattr(log_cfg, "max_file_size_mb", 50)
        defaults["max_files"] = getattr(log_cfg, "max_files", 10)
        # Override con config de módulo específico
        mod_cfg = getattr(log_cfg, section, None)
        if mod_cfg:
            defaults["level"] = getattr(mod_cfg, "level", defaults["level"])
    except Exception:
        pass
    return defaults


# Registro de loggers ya creados (evita duplicar handlers)
_loggers: dict = {}


def get_logger(module_name: str) -> logging.Logger:
    """
    Devuelve un logger configurado para el módulo indicado.

    El logger escribe en:
      - logs/<module_name>_<YYYY-MM-DD>.log  (RotatingFileHandler)
      - stdout (StreamHandler) — visible en consola y Blender terminal

    Args:
        module_name: Nombre del módulo. Usado como nombre del logger y del archivo.
                     Ejemplos: "blender", "dinov2", "pipeline", "yolo_cenital"

    Returns:
        logging.Logger configurado y listo para usar.
    """

    # Normalizar nombre
    name = module_name.lower().replace(" ", "_")

    # Reusar logger existente
    if name in _loggers:
        return _loggers[name]

    cfg_vals = _get_config(name)

    if not cfg_vals.get("enabled", True):
        # Logging deshabilitado → devolver logger nulo
        null_logger = logging.getLogger(f"legov.{name}.null")
        null_logger.addHandler(logging.NullHandler())
        _loggers[name] = null_logger
        return null_logger

    # Nivel de log
    level_str = cfg_vals.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    # Formato
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(f"legov.{name}")
    logger.setLevel(level)
    logger.propagate = False  # No propagar al root logger

    # ── Handler 1: Archivo rotativo ──────────────────────────────────────────
    log_dir = _get_log_dir()
    today = date.today().isoformat()
    log_file = os.path.join(log_dir, f"{name}_{today}.log")

    max_bytes = int(cfg_vals.get("max_file_size_mb", 50)) * 1024 * 1024
    backup_count = int(cfg_vals.get("max_files", 10))

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception as e:
        # En Blender el sistema de archivos puede tener restricciones
        print(f"[Logger Warning] No se pudo crear file handler para {log_file}: {e}")

    # ── Handler 2: Stdout (visible en Blender terminal y consola) ───────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    # Primer mensaje
    logger.info(f"Logger iniciado → {log_file}")

    _loggers[name] = logger
    return logger


def log_execution_header(logger: logging.Logger, script_name: str, **kwargs) -> None:
    """
    Escribe una cabecera de ejecución en el log con los parámetros principales.

    Args:
        logger:      Logger obtenido con get_logger()
        script_name: Nombre del script que arranca
        **kwargs:    Parámetros clave-valor a registrar en la cabecera
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info(f"INICIO: {script_name}")
    logger.info(f"Fecha/Hora: {date.today().isoformat()}")
    for k, v in kwargs.items():
        logger.info(f"  {k}: {v}")
    logger.info(sep)


def log_execution_footer(logger: logging.Logger, script_name: str,
                         duration_s: float = None, **kwargs) -> None:
    """
    Escribe un pie de ejecución con estadísticas finales.

    Args:
        logger:      Logger obtenido con get_logger()
        script_name: Nombre del script que termina
        duration_s:  Duración total en segundos (opcional)
        **kwargs:    Estadísticas adicionales clave-valor
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info(f"FIN: {script_name}")
    if duration_s is not None:
        m, s = divmod(int(duration_s), 60)
        logger.info(f"  Duración total: {m}m {s}s ({duration_s:.1f}s)")
    for k, v in kwargs.items():
        logger.info(f"  {k}: {v}")
    logger.info(sep)
