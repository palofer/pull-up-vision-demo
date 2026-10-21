"""Cálculos geométricos y clasificación básica de repeticiones."""

import csv
from pathlib import Path

import numpy as np


def calcular_angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Devuelve el ángulo ABC en grados; 180° corresponde a una línea recta."""
    vector_a = a - b
    vector_c = c - b
    divisor = np.linalg.norm(vector_a) * np.linalg.norm(vector_c)
    if divisor == 0:
        return None
    coseno = np.dot(vector_a, vector_c) / divisor
    return float(np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0))))


def calcular_angulos_codos(
    datos: list[dict], umbral_confianza: float = 0.5
) -> list[dict]:
    """Calcula ambos codos por fotograma, usando a la persona más confiable."""
    angulos = []
    lados = [("left", 5, 7, 9), ("right", 6, 8, 10)]
    for frame in datos:
        fila = {"frame": frame["frame"], "person_idx": ""}
        if frame["xy"].shape[0]:
            persona = int(np.argmax(frame["person_conf"]))
            fila["person_idx"] = persona
            xy, conf = frame["xy"][persona], frame["conf"][persona]
            for lado, hombro, codo, muneca in lados:
                confianza = float(np.min(conf[[hombro, codo, muneca]]))
                angulo = calcular_angulo(xy[hombro], xy[codo], xy[muneca])
                fila[f"{lado}_elbow_deg"] = (
                    angulo if confianza >= umbral_confianza and angulo is not None else ""
                )
                fila[f"{lado}_min_conf"] = confianza
        angulos.append(fila)
    return angulos


def calcular_holgura_cabeza(
    datos: list[dict], umbral_confianza: float = 0.3
) -> dict[int, float]:
    """Holgura cabeza–muñecas por fotograma (positivo = cabeza sobre las muñecas)."""
    holgura: dict[int, float] = {}
    for frame in datos:
        if frame["xy"].shape[0] == 0:
            continue
        persona = int(np.argmax(frame["person_conf"]))
        xy, conf = frame["xy"][persona], frame["conf"][persona]
        orejas = [xy[i] for i in (3, 4) if conf[i] >= umbral_confianza]
        munecas = [xy[i] for i in (9, 10) if conf[i] >= umbral_confianza]
        if not orejas or not munecas:
            continue
        y_orejas = float(np.mean([p[1] for p in orejas]))
        y_munecas = float(np.mean([p[1] for p in munecas]))
        holgura[int(frame["frame"])] = y_munecas - y_orejas
    return holgura


def detectar_repeticiones(
    angulos: list[dict],
    datos: list[dict],
    umbral_holgura: float = 20.0,
    umbral_extension: float = 150.0,
    umbral_inicio: float = 130.0,
) -> list[dict]:
    """Detecta ciclos por codos y los clasifica por holgura cabeza–muñecas."""
    holgura = calcular_holgura_cabeza(datos)
    repeticiones = []
    actual = None
    ultimo_frame = None
    for fila in angulos:
        frame = int(fila["frame"])
        if fila.get("left_elbow_deg", "") == "" or fila.get("right_elbow_deg", "") == "":
            continue
        izq, der = float(fila["left_elbow_deg"]), float(fila["right_elbow_deg"])
        extendido = izq >= umbral_extension and der >= umbral_extension
        inicio = min(izq, der) <= umbral_inicio
        if actual is None and not extendido and inicio:
            actual = {
                "inicio_frame": frame,
                "angulo_min_izq": izq,
                "angulo_min_der": der,
                "mejor_max": max(izq, der),
                "tope_frame": frame,
            }
        if actual is None:
            continue
        actual["angulo_min_izq"] = min(actual["angulo_min_izq"], izq)
        actual["angulo_min_der"] = min(actual["angulo_min_der"], der)
        if max(izq, der) < actual["mejor_max"]:
            actual["mejor_max"] = max(izq, der)
            actual["tope_frame"] = frame
        ultimo_frame = frame
        if extendido:
            actual["fin_frame"] = frame
            holgura_ciclo = [
                holgura[f] for f in range(actual["inicio_frame"], frame + 1) if f in holgura
            ]
            actual["holgura_max"] = max(holgura_ciclo) if holgura_ciclo else None
            actual["estado"] = (
                "completa"
                if actual["holgura_max"] is not None and actual["holgura_max"] >= umbral_holgura
                else "parcial"
            )
            repeticiones.append(actual)
            actual = None
    if actual is not None:
        actual["fin_frame"] = ultimo_frame
        holgura_ciclo = [
            holgura[f] for f in range(actual["inicio_frame"], ultimo_frame + 1) if f in holgura
        ]
        actual["holgura_max"] = max(holgura_ciclo) if holgura_ciclo else None
        actual["estado"] = "sin retorno a extensión"
        repeticiones.append(actual)
    return repeticiones


def guardar_csv_angulos(angulos: list[dict], ruta: Path) -> None:
    columnas = [
        "frame", "person_idx", "left_elbow_deg", "left_min_conf",
        "right_elbow_deg", "right_min_conf",
    ]
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(angulos)


def guardar_csv_repeticiones(repeticiones: list[dict], ruta: Path) -> None:
    columnas = [
        "inicio_frame", "tope_frame", "fin_frame", "angulo_min_izq",
        "angulo_min_der", "holgura_max", "estado",
    ]
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        for rep in repeticiones:
            holgura = rep.get("holgura_max")
            writer.writerow({
                "inicio_frame": rep["inicio_frame"],
                "tope_frame": rep["tope_frame"],
                "fin_frame": rep["fin_frame"],
                "angulo_min_izq": round(rep["angulo_min_izq"], 1),
                "angulo_min_der": round(rep["angulo_min_der"], 1),
                "holgura_max": round(holgura, 1) if holgura is not None else "",
                "estado": rep["estado"],
            })
