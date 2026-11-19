# Pull-Up Vision

Vocabulario de los ciclos de movimiento detectados en vídeos de dominadas y de las métricas que se calculan sobre ellos.

## Repeticiones y métricas

**Inicio extendido observado**:
La última posición extendida capturada antes de que comience la flexión de una repetición. Si el vídeo no la contiene, no se conoce el inicio necesario para medir su ROM ni el tiempo de subida.

**No extension**:
Estado de un ciclo detectado que no alcanza una posición extendida antes de que termine el vídeo. Es distinto de una repetición parcial: se cuenta como categoría propia, no como parcial.
_Avoid_: Sin retorno a extensión

**Pausa a mitad de repetición**:
Interrupción posible del movimiento durante una repetición; puede ser leve y no necesariamente un stop pronunciado.
_Avoid_: Stop in the middle

**Altura del agarre**:
Referencia vertical del agarre, estimada con la altura mediana de las muñecas.
_Avoid_: Altura de la barra, Barra detectada

**Altura de las orejas**:
Altura observada de las orejas, usada como aproximación de la barbilla porque el modelo no detecta directamente ese punto.
_Avoid_: Altura de la barbilla

**Altura de las orejas respecto al agarre**:
Distancia vertical entre las orejas y la referencia del agarre; es positiva cuando las orejas están por encima.
_Avoid_: Holgura cabeza–muñecas

**Superación del agarre**:
Posición en la que las orejas quedan al menos 20 px por encima de la referencia vertical del agarre.

**Extensión**:
Posición extendida de los brazos, determinada por el ángulo de los codos; es distinta del criterio para que una repetición sea completa.
_Avoid_: Extensión completa

**Repetición completa**:
Repetición en la que las orejas superan la referencia del agarre en algún momento del ciclo.

**Tiempo de subida**:
Tiempo desde el inicio extendido observado hasta el tope de la repetición. Está vacío si no se observó ese inicio.
_Avoid_: Pull

**Tiempo de bajada**:
Tiempo desde el tope de la repetición hasta el retorno a extensión. Está vacío si la repetición no retorna a extensión.
_Avoid_: Lower
