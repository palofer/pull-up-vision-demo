"""Cálculos geométricos y clasificación básica de repeticiones."""

import csv
from pathlib import Path

import numpy as np

UMBRAL_HOLGURA: float = 20.0
ROM_ANGULO_EXTENSION_REFERENCIA: float = 180.0
ROM_ANGULO_FLEXION_REFERENCIA: float = 55.0
UMBRAL_EXTENSION_CICLO: float = 150.0


def calcular_angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Devuelve el ángulo ABC en grados; 180° corresponde a una línea recta."""
    vector_a = a - b
    vector_c = c - b
    if np.linalg.norm(vector_a) == 0 or np.linalg.norm(vector_c) == 0:
        return None
    producto_cruzado = vector_a[0] * vector_c[1] - vector_a[1] * vector_c[0]
    producto_escalar = np.dot(vector_a, vector_c)
    return float(np.degrees(np.arctan2(abs(producto_cruzado), producto_escalar)))


def calcular_angulos_codos(datos: list[dict], umbral_confianza: float = 0.5) -> list[dict]:
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


def calcular_holgura_cabeza(datos: list[dict], umbral_confianza: float = 0.3) -> dict[int, float]:
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
    umbral_extension: float = UMBRAL_EXTENSION_CICLO,
    umbral_inicio: float = 130.0,
) -> list[dict]:
    """Detecta ciclos por codos y los clasifica por holgura cabeza–muñecas."""
    holgura = calcular_holgura_cabeza(datos)
    repeticiones = []
    actual = None
    ultimo_frame = None
    ultimo_frame_extendido = None
    for fila in angulos:
        frame = int(fila["frame"])
        if fila.get("left_elbow_deg", "") == "" or fila.get("right_elbow_deg", "") == "":
            continue
        izq, der = float(fila["left_elbow_deg"]), float(fila["right_elbow_deg"])
        extendido = izq >= umbral_extension and der >= umbral_extension
        inicio = min(izq, der) <= umbral_inicio
        if actual is None and extendido:
            ultimo_frame_extendido = frame
        if actual is None and not extendido and inicio:
            actual = {
                "inicio_frame": frame,
                "inicio_fase_frame": (
                    ultimo_frame_extendido if ultimo_frame_extendido is not None else frame
                ),
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
            ultimo_frame_extendido = frame
    if actual is not None:
        actual["fin_frame"] = ultimo_frame
        holgura_ciclo = [
            holgura[f] for f in range(actual["inicio_frame"], ultimo_frame + 1) if f in holgura
        ]
        actual["holgura_max"] = max(holgura_ciclo) if holgura_ciclo else None
        actual["estado"] = "sin retorno a extensión"
        repeticiones.append(actual)
    return repeticiones


def agregar_metricas_repeticiones(
    repeticiones: list[dict], angulos: list[dict], fps: float
) -> list[dict]:
    """Añade ROM porcentual y tiempos de subida/bajada a cada ciclo detectado."""
    if fps <= 0:
        raise ValueError("Los FPS deben ser mayores que cero")

    rango_referencia = ROM_ANGULO_EXTENSION_REFERENCIA - ROM_ANGULO_FLEXION_REFERENCIA
    enriquecidas = []
    for repeticion in repeticiones:
        inicio = int(repeticion.get("inicio_fase_frame", repeticion["inicio_frame"]))
        fin = int(repeticion["fin_frame"])
        filas_ciclo = [fila for fila in angulos if inicio <= int(fila["frame"]) <= fin]
        rom_por_lado = {}
        for lado in ("left", "right"):
            valores = [
                float(fila[f"{lado}_elbow_deg"])
                for fila in filas_ciclo
                if fila.get(f"{lado}_elbow_deg", "") != ""
            ]
            if valores:
                rom_por_lado[lado] = (max(valores) - min(valores)) / rango_referencia * 100

        rom_pct = (
            (rom_por_lado["left"] + rom_por_lado["right"]) / 2 if len(rom_por_lado) == 2 else ""
        )
        inicio_tope = (int(repeticion["tope_frame"]) - inicio) / fps
        fin_tope = (fin - int(repeticion["tope_frame"])) / fps
        enriquecida = dict(repeticion)
        enriquecida.update(
            {
                "rom_pct": rom_pct,
                "rom_izq_pct": rom_por_lado.get("left", ""),
                "rom_der_pct": rom_por_lado.get("right", ""),
                "tiempo_subida_s": inicio_tope,
                "tiempo_bajada_s": (
                    "" if repeticion["estado"] == "sin retorno a extensión" else fin_tope
                ),
            }
        )
        enriquecidas.append(enriquecida)
    return enriquecidas


def guardar_csv_angulos(angulos: list[dict], ruta: Path) -> None:
    columnas = [
        "frame",
        "person_idx",
        "left_elbow_deg",
        "left_min_conf",
        "right_elbow_deg",
        "right_min_conf",
    ]
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(angulos)


def guardar_csv_repeticiones(repeticiones: list[dict], ruta: Path) -> None:
    columnas = [
        "inicio_frame",
        "inicio_fase_frame",
        "tope_frame",
        "fin_frame",
        "angulo_min_izq",
        "angulo_min_der",
        "holgura_max",
        "estado",
        "rom_pct",
        "rom_izq_pct",
        "rom_der_pct",
        "tiempo_subida_s",
        "tiempo_bajada_s",
    ]
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        for rep in repeticiones:
            holgura = rep.get("holgura_max")
            writer.writerow(
                {
                    "inicio_frame": rep["inicio_frame"],
                    "inicio_fase_frame": rep.get("inicio_fase_frame", rep["inicio_frame"]),
                    "tope_frame": rep["tope_frame"],
                    "fin_frame": rep["fin_frame"],
                    "angulo_min_izq": round(rep["angulo_min_izq"], 1),
                    "angulo_min_der": round(rep["angulo_min_der"], 1),
                    "holgura_max": round(holgura, 1) if holgura is not None else "",
                    "estado": rep["estado"],
                    "rom_pct": round(rep["rom_pct"], 1) if rep["rom_pct"] != "" else "",
                    "rom_izq_pct": (
                        round(rep["rom_izq_pct"], 1) if rep["rom_izq_pct"] != "" else ""
                    ),
                    "rom_der_pct": (
                        round(rep["rom_der_pct"], 1) if rep["rom_der_pct"] != "" else ""
                    ),
                    "tiempo_subida_s": round(rep["tiempo_subida_s"], 2),
                    "tiempo_bajada_s": (
                        round(rep["tiempo_bajada_s"], 2) if rep["tiempo_bajada_s"] != "" else ""
                    ),
                }
            )


def guardar_csv_resumen(repeticiones: list[dict], video: str, ruta: Path) -> None:
    """Exporta una fila con los recuentos y promedios de un vídeo."""
    estados = ("completa", "parcial", "sin retorno a extensión")
    por_estado = {
        estado: sum(rep["estado"] == estado for rep in repeticiones) for estado in estados
    }
    repeticiones_con_retorno = [
        rep for rep in repeticiones if rep["estado"] in ("completa", "parcial")
    ]

    def promedio(campo: str, decimales: int) -> float | str:
        valores = [rep[campo] for rep in repeticiones_con_retorno if rep.get(campo, "") != ""]
        return round(sum(valores) / len(valores), decimales) if valores else ""

    columnas = [
        "video",
        "repeticiones_total",
        "repeticiones_completas",
        "repeticiones_parciales",
        "repeticiones_sin_retorno",
        "rom_pct_promedio",
        "tiempo_subida_s_promedio",
        "tiempo_bajada_s_promedio",
    ]
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        writer.writerow(
            {
                "video": video,
                "repeticiones_total": len(repeticiones),
                "repeticiones_completas": por_estado["completa"],
                "repeticiones_parciales": por_estado["parcial"],
                "repeticiones_sin_retorno": por_estado["sin retorno a extensión"],
                "rom_pct_promedio": promedio("rom_pct", 1),
                "tiempo_subida_s_promedio": promedio("tiempo_subida_s", 2),
                "tiempo_bajada_s_promedio": promedio("tiempo_bajada_s", 2),
            }
        )
