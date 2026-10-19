# Requisitos del proyecto

## Objetivo

Crear un prototipo reproducible que analice vídeos de dominadas, estime el movimiento a partir de keypoints y comunique resultados de forma visual. El sistema mide indicadores seleccionados; no debe presentarse como una evaluación médica ni como una valoración completa de la técnica.

## Convenciones de nombres

- Vídeos, resultados y documentos usan **kebab-case**: minúsculas y palabras separadas por guiones. Ejemplos: `dominadas-video-optimizado.mp4`, `project-requirements.md`.
- Los módulos Python usan nombres importables en minúsculas: `main.py`, `angles.py` y `visual.py`.
- Los resultados derivados se guardan en `data/processed/<video>/`: una carpeta por vídeo de entrada, con artefactos de nombre simple (`keypoints.csv`, `elbow-angles.png`…). Los vídeos de entrada permanecen en `data/raw/` y no se versionan.

## Estado actual

- El flujo usa YOLO-pose para analizar fotogramas y exporta keypoints a CSV.
- Calcula ángulos de codo y clasifica ciclos como completos o parciales mediante umbrales configurables en las funciones.
- Genera una interfaz de vídeo con keypoints 5–16, una gráfica de ángulos y un contador que avanza al terminar cada repetición.
- La interfaz recorta las bandas negras laterales detectadas en el vídeo de entrada.
- La configuración de dependencias usa `uv`, `pyproject.toml` y `uv.lock`.

## Requisitos de biomecánica

- [x] Calcular el ángulo del codo usando hombro, codo y muñeca.
- [ ] Usar `atan2` para el cálculo angular. Actualmente se usa producto escalar y `arccos`.
- [ ] Expresar el rango de movimiento (ROM) como porcentaje, además de la clasificación completa/parcial.
- [~] Graficar la diferencia vertical entre hombros; falta definir y validar el umbral de alerta.
- [ ] Exportar por separado el tiempo de subida (fase concéntrica) y bajada (fase excéntrica).
- [ ] Validar los umbrales con varios vídeos etiquetados manualmente; los valores actuales son experimentales, no estándares universales.

## Arquitectura y calidad del código

- [x] Separar la orquestación, los cálculos de ángulos/repeticiones y la visualización en `src/main.py`, `src/angles.py` y `src/visual.py`.
- [ ] Evitar números mágicos: mover umbrales como confianza `0.5`, flexión `55°`, extensión `150°` e inicio `130°` a una configuración central, por ejemplo `config.yaml`.
- [ ] Evaluar si separar la inferencia y el manejo de vídeo en módulos dedicados (`pose_extractor.py` y `video_processor.py`) aporta claridad sin sobreingeniería.
- [x] Tolerar fotogramas sin detecciones al calcular ángulos y al dibujar el HUD.
- [ ] Completar type hints y docstrings de todas las funciones públicas.

## Pruebas y ejecución

- [ ] Añadir pruebas unitarias de geometría; caso mínimo: puntos que formen 90° deben producir 90°.
- [ ] Añadir una CLI con `argparse` para seleccionar entrada y exportaciones.
- [x] Mantener dependencias declaradas y bloqueadas mediante `uv`.
- El comando de ejecución actual es `uv run python src/main.py`; aún no acepta argumentos.

## Exportación e interfaz

- [x] Exportar keypoints, ángulos y clasificación de cada repetición a CSV.
- [ ] Exportar un resumen agregado con repeticiones, ROM porcentual, tempo y alertas de asimetría.
- [x] Mostrar un HUD sin cajas delimitadoras, con líneas entre articulaciones y un panel lateral oscuro.
- [x] Actualizar el contador conforme terminan las repeticiones, sin mostrar resultados futuros durante la reproducción.
- [ ] Verificar legibilidad y rendimiento de la interfaz en vídeos de distintas resoluciones y orientaciones.
