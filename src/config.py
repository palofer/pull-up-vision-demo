"""Umbrales compartidos por el análisis y la visualización."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Configuracion:
    confianza_keypoints: float = 0.5
    confianza_holgura: float = 0.3
    umbral_holgura: float = 20.0
    umbral_extension: float = 150.0
    umbral_inicio: float = 130.0
    angulo_extension_referencia_rom: float = 180.0
    angulo_flexion_referencia_rom: float = 55.0


CONFIGURACION = Configuracion()
