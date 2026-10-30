import numpy as np

from src.angles import calcular_angulo


def test_angulo_recto():
    hombro = np.array([0, 1])
    codo = np.array([0, 0])
    muneca = np.array([1, 0])

    assert calcular_angulo(hombro, codo, muneca) == 90.0


def test_angulo_plano():
    hombro = np.array([1, 0])
    codo = np.array([0, 0])
    muneca = np.array([-1, 0])

    assert calcular_angulo(hombro, codo, muneca) == 180.0


def test_angulo_cero():
    hombro = np.array([1, 0])
    codo = np.array([0, 0])
    muneca = np.array([1, 0])

    assert calcular_angulo(hombro, codo, muneca) == 0.0
