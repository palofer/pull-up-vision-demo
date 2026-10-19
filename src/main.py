"""Ejecuta el análisis de pose de principio a fin."""

import csv
from pathlib import Path

from ultralytics import YOLO

from angles import (
    calcular_angulos_codos,
    detectar_repeticiones,
    guardar_csv_angulos,
    guardar_csv_repeticiones,
)
from visual import (
    guardar_grafica_angulos,
    guardar_grafica_hombros,
    guardar_interfaz,
)

ROOT = Path(__file__).resolve().parents[1]
VIDEO_POR_DEFECTO = ROOT / "data/raw/dominadas-video-optimizado.mp4"
PESOS_POR_DEFECTO = ROOT / "notebooks/yolo11s-pose.pt"


def guardar_csv_keypoints(datos: list[dict], ruta: Path) -> None:
    nombres = {
        0: "nariz", 1: "ojo_izq", 2: "ojo_der", 3: "oreja_izq", 4: "oreja_der",
        5: "hombro_izq", 6: "hombro_der", 7: "codo_izq", 8: "codo_der",
        9: "muneca_izq", 10: "muneca_der", 11: "cadera_izq", 12: "cadera_der",
        13: "rodilla_izq", 14: "rodilla_der", 15: "tobillo_izq", 16: "tobillo_der",
    }
    columnas = [
        "frame", "person_idx", "keypoint_idx", "keypoint", "x_px", "y_px",
        "keypoint_conf", "person_conf",
    ]
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        for frame in datos:
            for persona, keypoints in enumerate(frame["xy"]):
                for indice, (xy, confianza) in enumerate(
                    zip(keypoints, frame["conf"][persona], strict=True)
                ):
                    writer.writerow({
                        "frame": frame["frame"], "person_idx": persona,
                        "keypoint_idx": indice,
                        "keypoint": nombres.get(indice, f"keypoint_{indice}"),
                        "x_px": float(xy[0]), "y_px": float(xy[1]),
                        "keypoint_conf": float(confianza),
                        "person_conf": float(frame["person_conf"][persona]),
                    })


def ejecutar(
    video: Path = VIDEO_POR_DEFECTO,
    pesos: Path = PESOS_POR_DEFECTO,
) -> tuple[list[dict], list[dict], list[dict], dict[str, Path]]:
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    pesos = Path(pesos).expanduser()
    modelo = YOLO(str(pesos if pesos.is_file() else "yolo11s-pose.pt"))
    salida = ROOT / "data/processed" / video.stem
    salida.mkdir(parents=True, exist_ok=True)
    rutas = {
        "keypoints": salida / "keypoints.csv",
        "angulos": salida / "elbow-angles.csv",
        "repeticiones": salida / "repeticiones.csv",
        "grafica": salida / "elbow-angles.png",
        "hombros": salida / "shoulder-position.png",
        "interfaz": salida / "interfaz.mp4",
    }

    datos = []
    for frame_idx, resultado in enumerate(modelo(str(video), stream=True, verbose=False)):
        pose = resultado.keypoints
        datos.append({
            "frame": frame_idx,
            "xy": pose.xy.cpu().numpy(),
            "conf": pose.conf.cpu().numpy(),
            "person_conf": resultado.boxes.conf.cpu().numpy(),
        })
    guardar_csv_keypoints(datos, rutas["keypoints"])
    angulos = calcular_angulos_codos(datos)
    guardar_csv_angulos(angulos, rutas["angulos"])
    repeticiones = detectar_repeticiones(angulos)
    guardar_csv_repeticiones(repeticiones, rutas["repeticiones"])
    guardar_grafica_angulos(angulos, video, salida)
    guardar_grafica_hombros(angulos, rutas["keypoints"], video, salida)
    guardar_interfaz(video, rutas["keypoints"], angulos, repeticiones, rutas["interfaz"])
    return datos, angulos, repeticiones, rutas


if __name__ == "__main__":
    _, _, repeticiones, rutas = ejecutar()
    print(f"Repeticiones candidatas: {len(repeticiones)}")
    for ruta in rutas.values():
        print(f"Guardado: {ruta}")
