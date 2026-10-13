"""Gráficas y vídeo anotado para revisar las detecciones."""

import csv
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def guardar_grafica_angulos(angulos: list[dict], video: Path, salida: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError("No se pudieron leer los FPS del vídeo")
    tiempo = np.array([int(fila["frame"]) for fila in angulos]) / fps
    series = {}
    for lado, etiqueta in [("left", "Codo izquierdo"), ("right", "Codo derecho")]:
        series[etiqueta] = np.array([
            float(fila[f"{lado}_elbow_deg"])
            if fila.get(f"{lado}_elbow_deg", "") != "" else np.nan
            for fila in angulos
        ])
    fig, ax = plt.subplots(figsize=(12, 5))
    for etiqueta, valores in series.items():
        ax.plot(tiempo, valores, label=etiqueta)
    ax.set(xlabel="Tiempo (s)", ylabel="Ángulo del codo (°)", ylim=(180, 0))
    ax.set_title("Ángulo de los codos durante la dominada")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ruta = salida / f"{video.stem}-elbow-angles.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def guardar_video_anotado(
    video: Path,
    csv_keypoints: Path,
    angulos: list[dict],
    repeticiones: list[dict],
    salida: Path,
) -> Path:
    angulos_por_frame = {int(f["frame"]): f for f in angulos}
    persona_por_frame = {
        int(f["frame"]): int(f["person_idx"])
        for f in angulos if f.get("person_idx", "") != ""
    }
    puntos_por_frame = {}
    with csv_keypoints.open(newline="", encoding="utf-8") as archivo:
        for punto in csv.DictReader(archivo):
            frame = int(punto["frame"])
            if int(punto["person_idx"]) != persona_por_frame.get(frame, -1):
                continue
            puntos_por_frame.setdefault(frame, {})[int(punto["keypoint_idx"])] = (
                int(float(punto["x_px"])), int(float(punto["y_px"])),
                float(punto["keypoint_conf"]),
            )

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError("No se pudo abrir el vídeo o leer sus FPS")
    temp = salida.with_name(f"{salida.stem}-sin-audio.mp4")
    writer = cv2.VideoWriter(
        str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("No se pudo crear el vídeo temporal")

    conexiones = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
        (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    rangos = [
        (r["inicio_frame"], r["fin_frame"], i, r["estado"])
        for i, r in enumerate(repeticiones, start=1)
    ]
    frame_idx = 0
    while True:
        ok, imagen = cap.read()
        if not ok:
            break
        puntos = puntos_por_frame.get(frame_idx, {})
        for a, b in conexiones:
            if a in puntos and b in puntos and puntos[a][2] >= 0.5 and puntos[b][2] >= 0.5:
                cv2.line(imagen, puntos[a][:2], puntos[b][:2], (0, 220, 255), 3)
        for indice in range(5, 17):
            if indice in puntos:
                x, y, confianza = puntos[indice]
                color = (0, 200, 0) if confianza >= 0.5 else (0, 0, 255)
                cv2.circle(imagen, (x, y), 6, color, -1)
                cv2.putText(
                    imagen, str(indice), (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )
        fila = angulos_por_frame.get(frame_idx, {})
        izq, der = fila.get("left_elbow_deg", ""), fila.get("right_elbow_deg", "")
        etiqueta_angulos = (
            f"Codos: izq {float(izq):.0f} deg | der {float(der):.0f} deg"
            if izq and der else "Codos: sin deteccion"
        )
        rep_actual = next((r for r in rangos if r[0] <= frame_idx <= r[1]), None)
        etiqueta_rep = (
            f"Repeticion {rep_actual[2]}: {rep_actual[3].upper()}"
            if rep_actual else "Entre repeticiones"
        )
        cv2.putText(
            imagen, f"{frame_idx / fps:.2f} s | {etiqueta_angulos}", (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )
        color_rep = (0, 220, 0) if rep_actual and rep_actual[3] == "completa" else (0, 165, 255)
        cv2.putText(
            imagen, etiqueta_rep, (15, 58), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color_rep, 2,
        )
        writer.write(imagen)
        frame_idx += 1
    cap.release()
    writer.release()
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp), "-i", str(video), "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
        "-movflags", "+faststart", str(salida),
    ], check=True)
    temp.unlink()
    return salida
