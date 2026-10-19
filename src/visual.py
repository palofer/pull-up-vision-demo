"""Gráficas y vídeo anotado para revisar las detecciones."""

import csv
import subprocess
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


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


def guardar_grafica_hombros(
    angulos: list[dict], csv_keypoints: Path, video: Path, salida: Path
) -> Path:
    """Grafica la altura de ambos hombros y su diferencia vertical por fotograma."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError("No se pudieron leer los FPS del vídeo")

    persona_por_frame = {
        int(f["frame"]): int(f["person_idx"])
        for f in angulos if f.get("person_idx", "") != ""
    }
    frames = len(angulos)
    hombro_izq = np.full(frames, np.nan)
    hombro_der = np.full(frames, np.nan)
    with csv_keypoints.open(newline="", encoding="utf-8") as archivo:
        for punto in csv.DictReader(archivo):
            frame = int(punto["frame"])
            indice = int(punto["keypoint_idx"])
            if indice not in (5, 6) or frame >= frames:
                continue
            if int(punto["person_idx"]) != persona_por_frame.get(frame, -1):
                continue
            if float(punto["keypoint_conf"]) < 0.5:
                continue
            y = float(punto["y_px"])
            (hombro_izq if indice == 5 else hombro_der)[frame] = y

    tiempo = np.arange(frames) / fps
    diferencia = np.abs(hombro_izq - hombro_der)
    fig, (ax_pos, ax_dif) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    ax_pos.plot(tiempo, hombro_izq, label="Hombro izquierdo", color="tab:blue")
    ax_pos.plot(tiempo, hombro_der, label="Hombro derecho", color="tab:orange")
    ax_pos.set_ylabel("Coordenada vertical y (px)")
    ax_pos.set_title("Posición vertical de los hombros")
    ax_pos.invert_yaxis()
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend()
    ax_dif.plot(tiempo, diferencia, color="tab:purple", label="|Δy| entre hombros")
    ax_dif.fill_between(tiempo, diferencia, alpha=0.15, color="tab:purple")
    ax_dif.set(xlabel="Tiempo (s)", ylabel="Diferencia (px)")
    ax_dif.set_title("Asimetría vertical observada (sin umbral de alerta)")
    ax_dif.grid(True, alpha=0.3)
    ax_dif.legend()
    fig.tight_layout()
    ruta = salida / f"{video.stem}-shoulder-position.png"
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


def guardar_dashboard(
    video: Path,
    csv_keypoints: Path,
    angulos: list[dict],
    repeticiones: list[dict],
    salida: Path,
) -> Path:
    """Crea un vídeo dividido: persona a la izquierda, gráfica en vivo a la derecha."""
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

    rangos = [
        (r["inicio_frame"], r["fin_frame"], i, r["estado"])
        for i, r in enumerate(repeticiones, start=1)
    ]

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ancho_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError("No se pudo abrir el vídeo o leer sus FPS")

    ancho_grafica = 640
    alto_total = alto_video + 60
    ancho_total = ancho_video + ancho_grafica
    temp = salida.with_name(f"{salida.stem}-sin-audio.mp4")
    writer = cv2.VideoWriter(
        str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho_total, alto_total)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("No se pudo crear el vídeo temporal")

    # Prepara la figura de matplotlib con las series completas.
    tiempo = np.array([int(f["frame"]) for f in angulos]) / fps
    izq_vals = np.array([
        float(f["left_elbow_deg"]) if f.get("left_elbow_deg", "") != "" else np.nan
        for f in angulos
    ])
    der_vals = np.array([
        float(f["right_elbow_deg"]) if f.get("right_elbow_deg", "") != "" else np.nan
        for f in angulos
    ])
    fig, ax = plt.subplots(figsize=(ancho_grafica / 100, alto_video / 100), dpi=100)
    ax.plot(tiempo, izq_vals, label="Codo izquierdo", color="tab:blue")
    ax.plot(tiempo, der_vals, label="Codo derecho", color="tab:orange")
    ax.set(xlabel="Tiempo (s)", ylabel="Ángulo del codo (°)", ylim=(180, 0))
    ax.set_title("Ángulo de los codos")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    vline = ax.axvline(x=0, color="red", linestyle="--")
    dot_izq, = ax.plot([], [], "o", color="tab:blue")
    dot_der, = ax.plot([], [], "o", color="tab:orange")
    fig.tight_layout()

    conexiones = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
        (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    frame_idx = 0
    total_reps = len(repeticiones)
    completas = sum(1 for r in repeticiones if r["estado"] == "completa")
    parciales = total_reps - completas

    while True:
        ok, imagen = cap.read()
        if not ok:
            break

        # Dibuja keypoints 5-16.
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

        # Etiqueta de repetición actual.
        rep_actual = next((r for r in rangos if r[0] <= frame_idx <= r[1]), None)
        color_rep = (0, 220, 0) if rep_actual and rep_actual[3] == "completa" else (0, 165, 255)
        if rep_actual:
            etiqueta_rep = f"Rep {rep_actual[2]}/{total_reps}: {rep_actual[3].upper()}"
        elif frame_idx < (rangos[0][0] if rangos else 0):
            etiqueta_rep = "PREPARANDO..."
        elif frame_idx > (rangos[-1][1] if rangos else 0):
            etiqueta_rep = f"FINALIZADO: {completas} completas, {parciales} parciales"
        else:
            etiqueta_rep = "ENTRE REPETICIONES"
        cv2.putText(
            imagen, etiqueta_rep, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rep, 2,
        )

        # Actualiza la gráfica con el fotograma actual.
        tiempo_actual = frame_idx / fps
        fila = angulos_por_frame.get(frame_idx, {})
        izq = fila.get("left_elbow_deg", "")
        der = fila.get("right_elbow_deg", "")
        vline.set_xdata([tiempo_actual, tiempo_actual])
        if izq:
            dot_izq.set_data([tiempo_actual], [float(izq)])
        if der:
            dot_der.set_data([tiempo_actual], [float(der)])
        fig.canvas.draw()
        grafica = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        grafica_bgr = cv2.cvtColor(grafica, cv2.COLOR_RGB2BGR)

        # Monta el fotograma final con encabezado.
        lienzo = np.zeros((alto_total, ancho_total, 3), dtype=np.uint8)
        lienzo[60:60 + alto_video, 0:ancho_video] = imagen
        lienzo[60:60 + alto_video, ancho_video:ancho_total] = grafica_bgr
        cv2.rectangle(lienzo, (0, 0), (ancho_total, 60), (40, 40, 40), -1)
        cv2.putText(
            lienzo, etiqueta_rep, (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        angulos_txt = (
            f"Codos: izq {float(izq):.0f} deg | der {float(der):.0f} deg"
            if izq and der else "Codos: sin deteccion"
        )
        cv2.putText(
            lienzo, angulos_txt, (20, 54),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )
        cv2.putText(
            lienzo, f"{frame_idx / fps:.2f} s", (ancho_total - 140, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
        )

        writer.write(lienzo)
        frame_idx += 1

    cap.release()
    writer.release()
    plt.close(fig)

    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp), "-i", str(video), "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
        "-movflags", "+faststart", str(salida),
    ], check=True)
    temp.unlink()
    return salida


def guardar_interfaz(
    video: Path,
    csv_keypoints: Path,
    angulos: list[dict],
    repeticiones: list[dict],
    salida: Path,
) -> Path:
    """Genera un vídeo con la interfaz tipo app de entrenamiento."""
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
    alto_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError("No se pudo abrir el vídeo o leer sus FPS")

    ok, muestra = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("No se pudo leer el primer fotograma")
    gris = cv2.cvtColor(muestra, cv2.COLOR_BGR2GRAY)
    columnas_activas = np.flatnonzero(gris.mean(axis=0) > 8)
    if not columnas_activas.size:
        cap.release()
        raise RuntimeError("No se pudo detectar el área visible del vídeo")
    x_izq, x_der = int(columnas_activas[0]), int(columnas_activas[-1]) + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    alto_total = 1024
    escala = alto_total / alto_original
    ancho_video = round((x_der - x_izq) * escala)
    ancho_panel = 640
    ancho_total = ancho_video + ancho_panel
    temp = salida.with_name(f"{salida.stem}-sin-audio.mp4")
    writer = cv2.VideoWriter(
        str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho_total, alto_total)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("No se pudo crear el vídeo temporal")

    # Estadísticas por repetición.
    stats = []
    for i, rep in enumerate(repeticiones, start=1):
        inicio = int(rep["inicio_frame"])
        fin = int(rep["fin_frame"])
        frames_rep = [
            angulos_por_frame.get(f, {})
            for f in range(inicio, fin + 1)
            if f in angulos_por_frame
        ]
        izq_vals = [float(f["left_elbow_deg"]) for f in frames_rep if f.get("left_elbow_deg", "")]
        der_vals = [float(f["right_elbow_deg"]) for f in frames_rep if f.get("right_elbow_deg", "")]
        bottom = max(max(izq_vals, default=180), max(der_vals, default=180))
        top = min(min(izq_vals, default=0), min(der_vals, default=0))
        duracion = (int(rep["tope_frame"]) - inicio) / fps
        stats.append({
            "rep": i, "estado": rep["estado"],
            "inicio_frame": inicio, "fin_frame": fin,
            "bottom": bottom, "top": top, "pull": duracion,
        })

    # Curvas de ángulo del codo, con el eje invertido para mostrar la flexión hacia arriba.
    tiempos = np.array([int(f["frame"]) for f in angulos]) / fps
    angulo_izq = np.array([
        float(f["left_elbow_deg"]) if f.get("left_elbow_deg", "") != "" else np.nan
        for f in angulos
    ])
    angulo_der = np.array([
        float(f["right_elbow_deg"]) if f.get("right_elbow_deg", "") != "" else np.nan
        for f in angulos
    ])

    # Figura de matplotlib para la gráfica (estilo oscuro).
    ancho_graf = 6.0
    alto_graf = 4.4
    fig, ax = plt.subplots(figsize=(ancho_graf, alto_graf), dpi=100)
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    linea_izq, = ax.plot([], [], color="#2680eb", linewidth=2, label="Left elbow")
    linea_der, = ax.plot([], [], color="#ff8c32", linewidth=2, label="Right elbow")
    ax.axhline(y=55, color="#64cf85", linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(y=150, color="#888888", linestyle="--", linewidth=1, alpha=0.8)
    ax.set(
        xlabel="Time (s)", ylabel="Elbow angle (°)", ylim=(180, 0),
        xlim=(0, len(angulos) / fps),
    )
    ax.set_title("Elbow angle", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.grid(True, alpha=0.2, color="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.legend(loc="upper right", fontsize=8)
    vline = ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    dot_izq, = ax.plot([], [], "o", color="#2680eb", markersize=5)
    dot_der, = ax.plot([], [], "o", color="#ff8c32", markersize=5)
    fig.tight_layout()
    siguiente_zona = 0

    # Colores del esqueleto.
    colores_esqueleto = {
        (5, 7): (255, 255, 0), (7, 9): (255, 255, 0),  # brazo izq amarillo/cian
        (6, 8): (0, 165, 255), (8, 10): (0, 165, 255),  # brazo der naranja/azul
        (5, 6): (255, 255, 255), (5, 11): (255, 255, 255),
        (6, 12): (255, 255, 255), (11, 12): (255, 255, 255),  # torso blanco
        (11, 13): (255, 105, 180), (13, 15): (255, 105, 180),  # pierna izq rosa
        (12, 14): (0, 0, 255), (14, 16): (0, 0, 255),  # pierna der rojo
    }

    frame_idx = 0
    while True:
        ok, imagen = cap.read()
        if not ok:
            break
        imagen = imagen[:, x_izq:x_der]
        imagen = cv2.resize(imagen, (ancho_video, alto_total))

        # Dibuja el esqueleto con colores por región.
        puntos = puntos_por_frame.get(frame_idx, {})
        for (a, b), color in colores_esqueleto.items():
            if a in puntos and b in puntos and puntos[a][2] >= 0.5 and puntos[b][2] >= 0.5:
                inicio = (round((puntos[a][0] - x_izq) * escala), round(puntos[a][1] * escala))
                fin = (round((puntos[b][0] - x_izq) * escala), round(puntos[b][1] * escala))
                cv2.line(imagen, inicio, fin, color, 4)
        for indice in range(5, 17):
            if indice in puntos:
                x_original, y_original, confianza = puntos[indice]
                if not x_izq <= x_original < x_der:
                    continue
                x = round((x_original - x_izq) * escala)
                y = round(y_original * escala)
                color = (255, 255, 255) if confianza >= 0.5 else (0, 0, 255)
                cv2.circle(imagen, (x, y), 5, color, -1)
                cv2.circle(imagen, (x, y), 5, (0, 0, 0), 1)

        # Panel derecho oscuro.
        panel = np.full((alto_total, ancho_panel, 3), (25, 25, 28), dtype=np.uint8)

        # Métricas superiores.
        completadas = [s for s in stats if frame_idx >= s["fin_frame"]]
        completas = sum(1 for s in completadas if s["estado"] == "completa")
        cv2.putText(panel, f"Reps: {len(completadas)}", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        color_rom = (
            (0, 255, 128)
            if completadas and completas == len(completadas)
            else (0, 165, 255)
        )
        cv2.putText(panel, f"Full ROM: {completas} / {len(completadas)}", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rom, 2)

        fila = angulos_por_frame.get(frame_idx, {})
        izq, der = fila.get("left_elbow_deg", ""), fila.get("right_elbow_deg", "")
        elbow_txt = (
            f"Elbow {((float(izq) + float(der)) / 2):.0f}\u00b0"
            if izq and der else "Elbow --"
        )
        cv2.putText(panel, elbow_txt, (ancho_panel - 210, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Actualiza y coloca la gráfica.
        tiempo_actual = frame_idx / fps
        hasta_ahora = min(frame_idx + 1, len(tiempos))
        linea_izq.set_data(tiempos[:hasta_ahora], angulo_izq[:hasta_ahora])
        linea_der.set_data(tiempos[:hasta_ahora], angulo_der[:hasta_ahora])
        while siguiente_zona < len(stats) and frame_idx >= stats[siguiente_zona]["fin_frame"]:
            rep = stats[siguiente_zona]
            color = "#2e8b57" if rep["estado"] == "completa" else "#8b5a2b"
            ax.axvspan(
                rep["inicio_frame"] / fps, rep["fin_frame"] / fps,
                color=color, alpha=0.25,
            )
            ax.text(
                (rep["inicio_frame"] + rep["fin_frame"]) / (2 * fps), 8,
                str(rep["rep"]), color="white", ha="center", fontsize=9,
                fontweight="bold",
            )
            siguiente_zona += 1
        if izq:
            dot_izq.set_data([tiempo_actual], [float(izq)])
        else:
            dot_izq.set_data([], [])
        if der:
            dot_der.set_data([tiempo_actual], [float(der)])
        else:
            dot_der.set_data([], [])
        vline.set_xdata([tiempo_actual, tiempo_actual])
        fig.canvas.draw()
        grafica = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        grafica_bgr = cv2.cvtColor(grafica, cv2.COLOR_RGB2BGR)
        alto_graf_px, ancho_graf_px = grafica_bgr.shape[:2]
        y_graf = 155
        panel[y_graf:y_graf + alto_graf_px, 40:40 + ancho_graf_px] = grafica_bgr

        # Tabla de repeticiones.
        y_tabla = y_graf + alto_graf_px + 40
        columnas = ["REP", "ROM", "BOTTOM", "TOP", "PULL"]
        xs = [50, 150, 240, 340, 440]
        for x, col in zip(xs, columnas, strict=False):
            cv2.putText(panel, col, (x, y_tabla), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (160, 160, 160), 1)
        for i, s in enumerate(completadas):
            y = y_tabla + 35 * (i + 1)
            color_estado = (0, 255, 128) if s["estado"] == "completa" else (0, 165, 255)
            rom_txt = "FULL" if s["estado"] == "completa" else "PARTIAL"
            fila_txt = [
                str(s["rep"]), rom_txt, f"{s['bottom']:.0f}\u00b0",
                f"{s['top']:.0f}\u00b0", f"{s['pull']:.2f}s",
            ]
            for x, txt, color in zip(xs, fila_txt,
                                     [(255, 255, 255), color_estado,
                                      (255, 255, 255), (255, 255, 255),
                                      (255, 255, 255)], strict=False):
                cv2.putText(panel, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 1)

        lienzo = np.zeros((alto_total, ancho_total, 3), dtype=np.uint8)
        lienzo[:, :ancho_video] = imagen
        lienzo[:, ancho_video:] = panel
        writer.write(lienzo)
        frame_idx += 1

    cap.release()
    writer.release()
    plt.close(fig)

    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp), "-i", str(video), "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
        "-movflags", "+faststart", str(salida),
    ], check=True)
    temp.unlink()
    return salida
