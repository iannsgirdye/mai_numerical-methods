"""Задание 1.2. Метод прогонки для трёхдиагональной системы."""

from pathlib import Path

import numpy as np

# Настройки вывода и порог, ниже которого знаменатель считаем нулевым.
ARRAY_DECIMAL_PLACES = 6
ARRAY_LINE_WIDTH = 120
DENOMINATOR_TOLERANCE = 1e-12

np.set_printoptions(precision=ARRAY_DECIMAL_PLACES, suppress=False, linewidth=ARRAY_LINE_WIDTH)


def read_input():
    with Path(__file__).with_name("input.txt").open(encoding="utf-8") as file:
        return [[float(value) for value in line.split()] for line in file if line.strip()]


def thomas(lower, diagonal, upper, b):
    """Решить трёхдиагональную систему методом прогонки.

    lower    — нижняя диагональ (n-1 элемент).
    diagonal — главная диагональ (n элементов).
    upper    — верхняя диагональ (n-1 элемент).
    b        — вектор правых частей (n элементов).
    P и Q    — прогоночные коэффициенты.
    """
    n = len(diagonal)
    P = np.zeros(n)
    Q = np.zeros(n)
    x = np.zeros(n)

    # На первом шаге нет элемента нижней диагонали.
    first_row_scale = max(abs(diagonal[0]), abs(upper[0]) if n > 1 else 0.0)
    if abs(diagonal[0]) <= DENOMINATOR_TOLERANCE * first_row_scale:
        raise ValueError("Нулевой знаменатель на первом шаге метода прогонки.")
    if n > 1:
        P[0] = -upper[0] / diagonal[0]
    Q[0] = b[0] / diagonal[0]

    # Прямой ход: вычисляем P_i и Q_i для следующих уравнений.
    for i in range(1, n):
        denominator = diagonal[i] + lower[i - 1] * P[i - 1]
        row_scale = max(abs(diagonal[i]), abs(lower[i - 1]), abs(upper[i]) if i < n - 1 else 0.0)
        if abs(denominator) <= DENOMINATOR_TOLERANCE * row_scale:
            raise ValueError(f"Нулевой знаменатель на шаге {i + 1} метода прогонки.")
        if i < n - 1:
            P[i] = -upper[i] / denominator
        Q[i] = (b[i] - lower[i - 1] * Q[i - 1]) / denominator

    # У последнего уравнения нет элемента верхней диагонали.
    P[-1] = 0.0

    # Обратный ход: используем x_i = P_i · x_(i+1) + Q_i.
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return P, Q, x


def main():
    data = read_input()
    n = int(data[0][0])

    # Считываем только диагонали и правую часть; полную матрицу не создаём.
    # При n = 1 нижняя и верхняя диагонали пусты, поэтому строки в файле опускаются.
    if n == 1:
        lower = np.array([], dtype=float)
        diagonal = np.array(data[1], dtype=float)
        upper = np.array([], dtype=float)
        b = np.array(data[2], dtype=float)
    else:
        lower, diagonal, upper, b = [np.array(row, dtype=float) for row in data[1:5]]

    P, Q, x = thomas(lower, diagonal, upper, b)

    print(f"Прогоночные коэффициенты P:\n{P}\n")
    print(f"Прогоночные коэффициенты Q:\n{Q}\n")
    print(f"Решение системы x:\n{x}\n")


if __name__ == "__main__":
    main()
