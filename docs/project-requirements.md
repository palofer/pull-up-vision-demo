# Requisitos del proyecto

## Objetivo

Crear un prototipo reproducible que analice vídeos de dominadas, estime el movimiento a partir de keypoints y comunique resultados de forma visual. El sistema mide indicadores seleccionados; no debe presentarse como una evaluación médica ni como una valoración completa de la técnica.

## Convenciones de nombres

- Vídeos, resultados y documentos usan **kebab-case**: minúsculas y palabras separadas por guiones. Ejemplos: `dominadas-video-optimizado.mp4`, `project-requirements.md`.
- Los módulos Python usan nombres importables en minúsculas: `main.py`, `angles.py`, `config.py`, `pose.py`, `visual.py`, `render.py` y `viewer.py`.
- Los resultados derivados se guardan en `data/processed/<video>/`: una carpeta por vídeo de entrada, con artefactos de nombre simple (`keypoints.csv`, `viewer.html`…). Los vídeos de entrada permanecen en `data/raw/` y no se versionan.

## Estado actual

- El flujo usa YOLO-pose para analizar fotogramas y exporta keypoints a CSV.
- Calcula ángulos de codo y posiciones de hombros como datos de técnica/asimetría, y clasifica cada ciclo por la altura de las orejas respecto a una referencia fija del agarre.
- Un ciclo que no retorna a extensión antes de terminar el vídeo se clasifica como `no_extension`.
- Los estados `completa`, `parcial` y `no_extension` son categorías separadas en el resumen y en la tabla del visor. `no_extension` cuenta en el total, pero no como parcial.
- Genera un visor HTML de un solo archivo con el vídeo original (referenciado por ruta relativa), keypoints 5–16 dibujados encima, gráficas sincronizadas de altura de orejas, ángulo de codos y posición de hombros, y una tabla de repeticiones.
- El visor recorta las franjas negras laterales del vídeo original sin modificar el archivo y lo referencia mediante una ruta relativa desde la carpeta de resultados.
- La configuración de dependencias usa `uv`, `pyproject.toml` y `uv.lock`.

## Requisitos de biomecánica

- [x] Calcular el ángulo del codo usando hombro, codo y muñeca.
- [x] Usar la altura de las orejas respecto a la referencia mediana de las muñecas (máxima del ciclo ≥ 20 px) como criterio de completa/parcial; conservar los ángulos de codo como dato de técnica y referencia visual de extensión.
- [x] Usar `atan2` para el cálculo angular.
- [x] Expresar el rango de movimiento (ROM) por brazo y como promedio bilateral, además de la clasificación completa/parcial; medir desde el último fotograma extendido observado y usar 180°–55° como referencia provisional, sin limitar el resultado a 100 %. Si no se observó ese inicio, dejar el ROM vacío en vez de estimarlo.
- [~] Graficar la diferencia vertical entre hombros; el visor ya muestra la posición de cada hombro y `shoulder-position.png` la diferencia, pero falta definir y validar el umbral de alerta.
- [x] Exportar por separado el tiempo de subida (desde el último fotograma extendido observado hasta el tope) y bajada (del tope al retorno a extensión). Dejar la subida vacía si falta el inicio observado y la bajada si no hay retorno.
- [x] Validar los umbrales con varios vídeos etiquetados manualmente; los valores actuales son experimentales, no estándares universales. Evidencia: `data/labels/manual-repetition-labels.csv` (34 repeticiones etiquetadas en 6 grabaciones).

## Arquitectura y calidad del código

- [x] Separar la orquestación, los cálculos de ángulos/repeticiones y la visualización en `src/main.py`, `src/angles.py`, `src/visual.py` (gráficas estáticas) y `src/render.py` (vídeo MP4).
- [x] Evitar números mágicos: centralizar confianza `0.5`, flexión `55°`, extensión `150°`, inicio `130°` y los demás umbrales de análisis en `src/config.py`, y las constantes de layout en `VisualConfig`.
- [ ] Evaluar si separar la inferencia y el manejo de vídeo en módulos dedicados (`pose_extractor.py` y `video_processor.py`) aporta claridad sin sobreingeniería.
- [x] Tolerar fotogramas sin detecciones al calcular ángulos y al dibujar el HUD.
- [x] Completar type hints y docstrings de todas las funciones públicas.

## Pruebas y ejecución

- [x] Añadir pruebas unitarias de geometría; caso mínimo: puntos que formen 90° deben producir 90°.
- [x] Añadir una CLI con `argparse` para elegir el vídeo de entrada, los pesos y la exportación MP4 (`--export-mp4`).
- [x] Mantener dependencias declaradas y bloqueadas mediante `uv`.
- El comando de ejecución es `uv run python src/main.py`.

## Exportación e interfaz

- [x] Exportar keypoints, ángulos y clasificación de cada repetición a CSV.
- [x] Exportar un resumen agregado con recuentos de repeticiones, ROM porcentual y tempo.
- [ ] Exportar alertas de asimetría tras definir y validar el umbral.
- [x] Mostrar un visor con esqueleto sin cajas delimitadoras, una gráfica de orejas con línea de umbral de superación, gráficas de codos y hombros a mitad de ancho, y una tabla de repeticiones; todo sin scroll en 1440×800 px.
- [x] Sincronizar vídeo, esqueleto y gráficas; actualizar la tabla y sombrear repeticiones solo cuando terminan, sin revelar resultados futuros.
- [ ] Verificar legibilidad y rendimiento de la interfaz en vídeos de distintas resoluciones y orientaciones.
