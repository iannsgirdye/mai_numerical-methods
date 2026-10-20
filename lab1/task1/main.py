"""Задание 1.1. LU-разложение с выбором главного элемента."""

from pathlib import Path

import numpy as np

# Настройки отображения результатов и числовой порог для ведущего элемента.
ARRAY_DECIMAL_PLACES = 6
ARRAY_LINE_WIDTH = 120
PIVOT_TOLERANCE = 1e-12
COMPARISON_SCIENTIFIC_DIGITS = 3
DETERMINANT_SIGNIFICANT_DIGITS = 10

np.set_printoptions(precision=ARRAY_DECIMAL_PLACES, suppress=False, linewidth=ARRAY_LINE_WIDTH)


def read_input():
    """Прочитать числа из input.txt рядом с этим скриптом."""
    with Path(__file__).with_name("input.txt").open(encoding="utf-8") as file:
        return [[float(value) for value in line.split()] for line in file if line.strip()]


def max_abs_difference(first, second):
    """Найти самое большое по модулю расхождение между двумя результатами."""
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def lu(A):
    """Разложить A на матрицы P, L и U, чтобы P · A = L · U.

    P — матрица перестановок строк.
    L — нижняя треугольная матрица.
    U — верхняя треугольная матрица.
    Выбор главного элемента помогает избежать деления на слишком маленькое число.
    """
    n = len(A)
    U = A.copy().astype(float)  # Рабочая копия A; во время вычислений она станет U.
    L = np.eye(n)               # L начинаем с единичной матрицы.
    P = np.eye(n)               # P тоже сначала единичная: перестановок ещё не было.
    swaps = 0                   # Считаем перестановки строк для вычисления определителя.

    for k in range(n):
        # Ищем в текущем столбце элемент с наибольшим модулем ниже диагонали.
        # Его строка станет ведущей (опорной) на этом шаге.
        row = k + np.argmax(np.abs(U[k:, k]))
        pivot_row_scale = float(np.max(np.abs(U[row, k:])))
        pivot_tolerance = PIVOT_TOLERANCE * pivot_row_scale
        if abs(U[row, k]) <= pivot_tolerance:
            raise ValueError("Матрица вырождена или численно плохо обусловлена: не найден надёжный опорный элемент.")

        if row != k:
            # Меняем строки местами в U и P.
            U[[k, row]] = U[[row, k]]
            P[[k, row]] = P[[row, k]]
            # Уже найденные элементы L в этих строках тоже должны поменяться.
            if k:
                L[[k, row], :k] = L[[row, k], :k]
            swaps += 1

        # Обнуляем элементы под диагональю текущего столбца.
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]       # Множитель исключения неизвестной.
            U[i, k:] -= L[i, k] * U[k, k:]   # Вычитаем нужное кратное ведущей строки.

    return L, U, P, swaps


def solve_lu(L, U, P, b):
    """Решить систему A · x = b, используя уже найденные L, U и P.

    Сначала прямой подстановкой решаем L · y = P · b,
    затем обратной подстановкой решаем U · x = y.
    """
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)
    pb = P @ b  # Переставляем правую часть так же, как строки матрицы A.

    # Прямая подстановка: вычисляем y слева направо.
    for i in range(n):
        y[i] = (pb[i] - L[i, :i] @ y[:i]) / L[i, i]

    # Обратная подстановка: вычисляем x справа налево.
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    return x


def main():
    # Читаем input.txt, который лежит в этой же папке с main.py.
    data = read_input()
    n = int(data[0][0])

    # Первые n столбцов расширенной матрицы — коэффициенты A, последний — правая часть b.
    Augmented = np.array(data[1:n + 1], dtype=float)
    A = Augmented[:, :n]
    b = Augmented[:, n]

    L, U, P, swaps = lu(A)
    x = solve_lu(L, U, P, b)

    # Обратная матрица: решаем A · x = e для каждого столбца единичной матрицы E.
    E = np.eye(n)
    A_inverse = np.column_stack([solve_lu(L, U, P, E[:, j]) for j in range(n)])

    # Определитель U равен произведению его диагональных элементов.
    # Каждая перестановка строк меняет знак определителя.
    determinant = (-1) ** swaps * np.prod(np.diag(U))
    LU_product = L @ U
    P_A_product = P @ A
    A_A_inverse_product = A @ A_inverse

    print(f"Исходная матрица A:\n{A}\n")
    print(f"Матрица L:\n{L}\n")
    print(f"Матрица U:\n{U}\n")
    print(f"Матрица перестановок P:\n{P}\n")
    print(f"Произведение L · U:\n{LU_product}\n")
    print(f"Произведение P · A:\n{P_A_product}\n")
    print(f"Проверка P · A = L · U (максимальное расхождение):\n{max_abs_difference(P_A_product, LU_product):.{COMPARISON_SCIENTIFIC_DIGITS}e}\n")
    print(f"Решение системы x:\n{x}\n")
    print(f"Обратная матрица A⁻¹:\n{A_inverse}\n")
    print(f"Определитель det(A):\n{determinant:.{DETERMINANT_SIGNIFICANT_DIGITS}g}\n")
    print(f"Произведение A · A⁻¹ (должна получиться единичная матрица):\n{A_A_inverse_product}\n")
    print(f"Проверка A · A⁻¹ = E (максимальное расхождение):\n{max_abs_difference(A_A_inverse_product, E):.{COMPARISON_SCIENTIFIC_DIGITS}e}\n")


if __name__ == "__main__":
    main()
