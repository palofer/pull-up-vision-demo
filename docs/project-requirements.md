# Requisitos del proyecto

## Objetivo

Crear un prototipo reproducible que analice vídeos de dominadas, estime el movimiento a partir de keypoints y comunique resultados de forma visual. El sistema mide indicadores seleccionados; no debe presentarse como una evaluación médica ni como una valoración completa de la técnica.

## Convenciones de nombres

- Vídeos, resultados y documentos usan **kebab-case**: minúsculas y palabras separadas por guiones. Ejemplos: `dominadas-video-optimizado.mp4`, `project-requirements.md`.
- Los módulos Python usan nombres importables en minúsculas: `main.py`, `angles.py` y `visual.py`.
- Los resultados derivados se guardan en `data/processed/<video>/`: una carpeta por vídeo de entrada, con artefactos de nombre simple (`keypoints.csv`, `elbow-angles.png`…). Los vídeos de entrada permanecen en `data/raw/` y no se versionan.

## Estado actual

- El flujo usa YOLO-pose para analizar fotogramas y exporta keypoints a CSV.
- Calcula ángulos de codo y posiciones de hombros como datos de técnica/asimetría, y clasifica cada ciclo como completo o parcial mediante la holgura cabeza–muñecas (criterio independiente del agarre).
- Un ciclo que no retorna a extensión antes de terminar el vídeo se clasifica como `no_extension`.
- Los recuentos de interfaz tratan `completa`, `parcial` y `no_extension` como categorías separadas. `no_extension` cuenta en el total, pero no como parcial; el porcentaje Full ROM es el número de completas dividido por todas las repeticiones detectadas.
- Genera una interfaz de vídeo con keypoints 5–16, una gráfica de ángulos y un contador que avanza al terminar cada repetición.
- La interfaz recorta las bandas negras laterales detectadas en el vídeo de entrada.
- La configuración de dependencias usa `uv`, `pyproject.toml` y `uv.lock`.

## Requisitos de biomecánica

- [x] Calcular el ángulo del codo usando hombro, codo y muñeca.
- [x] Usar la holgura cabeza–muñecas (máxima del ciclo ≥ 20 px) como criterio de completa/parcial, independiente del agarre; los ángulos de codo se conservan solo como dato de técnica.
- [x] Usar `atan2` para el cálculo angular.
- [x] Expresar el rango de movimiento (ROM) por brazo y como promedio bilateral, además de la clasificación completa/parcial; medir desde el último fotograma extendido observado y usar 180°–55° como referencia provisional, sin limitar el resultado a 100 %. Si no se observó ese inicio, dejar el ROM vacío en vez de estimarlo.
- [~] Graficar la diferencia vertical entre hombros; falta definir y validar el umbral de alerta.
- [x] Exportar por separado el tiempo de subida (desde el último fotograma extendido observado hasta el tope) y bajada (del tope al retorno a extensión). Dejar la subida vacía si falta el inicio observado y la bajada si no hay retorno.
- [ ] Validar los umbrales con varios vídeos etiquetados manualmente; los valores actuales son experimentales, no estándares universales.

## Arquitectura y calidad del código

- [x] Separar la orquestación, los cálculos de ángulos/repeticiones y la visualización en `src/main.py`, `src/angles.py` y `src/visual.py`.
- [x] Evitar números mágicos: centralizar confianza `0.5`, flexión `55°`, extensión `150°`, inicio `130°` y los demás umbrales de análisis en `src/config.py`.
- [ ] Evaluar si separar la inferencia y el manejo de vídeo en módulos dedicados (`pose_extractor.py` y `video_processor.py`) aporta claridad sin sobreingeniería.
- [x] Tolerar fotogramas sin detecciones al calcular ángulos y al dibujar el HUD.
- [ ] Completar type hints y docstrings de todas las funciones públicas.

## Pruebas y ejecución

- [x] Añadir pruebas unitarias de geometría; caso mínimo: puntos que formen 90° deben producir 90°.
- [ ] Añadir una CLI con `argparse` para seleccionar entrada y exportaciones.
- [x] Mantener dependencias declaradas y bloqueadas mediante `uv`.
- El comando de ejecución actual es `uv run python src/main.py`; aún no acepta argumentos.

## Exportación e interfaz

- [x] Exportar keypoints, ángulos y clasificación de cada repetición a CSV.
- [x] Exportar un resumen agregado con recuentos de repeticiones, ROM porcentual y tempo.
- [ ] Exportar alertas de asimetría tras definir y validar el umbral.
- [x] Mostrar un HUD sin cajas delimitadoras, con líneas entre articulaciones y un panel lateral oscuro.
- [x] Actualizar el contador conforme terminan las repeticiones, sin mostrar resultados futuros durante la reproducción.
- [ ] Verificar legibilidad y rendimiento de la interfaz en vídeos de distintas resoluciones y orientaciones.
