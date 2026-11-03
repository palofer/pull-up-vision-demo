import numpy as np
import pytest

from src.angles import calcular_angulo


def test_angulo_recto():
    hombro = np.array([0, 1])
    codo = np.array([0, 0])
    muneca = np.array([1, 0])

    assert calcular_angulo(hombro, codo, muneca) == pytest.approx(90.0)
