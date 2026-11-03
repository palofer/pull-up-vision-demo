import pytest
import numpy as np
from src.angles import calcular_holgura_cabeza


@pytest.fixture
def mock_pose_data():
    """
    Fixture generador para inyectar datos de pose mínimos.
    Crea la estructura base de 17 puntos y solo sobreescribe los solicitados.
    """

    def _generador(puntos_relevantes, frame_id=0):
        # 1. Inicializamos arrays de 17 puntos con ceros (minimal mocking)
        xy = np.zeros((1, 17, 2), dtype=np.float32)
        conf = np.zeros((1, 17), dtype=np.float32)

        # 2. Asignamos solo los puntos que nos importan para el test
        for idx, (coordenadas, confianza) in puntos_relevantes.items():
            xy[0, idx] = coordenadas
            conf[0, idx] = confianza

        # 3. Construimos y devolvemos la estructura masiva limpia
        return [
            {
                "frame": frame_id,
                "xy": xy,
                "conf": conf,
                "person_conf": np.array([0.90], dtype=np.float32),
            }
        ]

    return _generador


def test_holgura_cabeza(mock_pose_data):
    puntos_de_prueba = {
        3: ([320, 80], 0.98),  # Oreja Izquierda
        4: ([300, 80], 0.97),  # Oreja Derecha
        9: ([290, 130], 0.93),  # Muñeca Izquierdo
        10: ([350, 130], 0.94),  # Muñeca Derecho
    }

    datos = mock_pose_data(puntos_de_prueba, frame_id=0)

    resultado = calcular_holgura_cabeza(datos)

    assert resultado == {0: pytest.approx(172.0)}


def test_holgura_cabeza_con_confianza_baja(mock_pose_data):
    puntos_de_prueba = {
        3: ([320, 80], 0.98),  # Oreja Izquierda
        4: ([300, 80], 0.97),  # Oreja Derecha
        9: ([290, 130], 0.93),  # Muñeca Izquierdo
        10: ([350, 130], 0.94),  # Muñeca Derecho
    }

    datos = mock_pose_data(puntos_de_prueba)
    datos[0]["conf"][0][3] = 0.4

    with pytest.raises(ValueError, match="Confianza de keypoints insuficiente"):
        calcular_holgura_cabeza(datos)
