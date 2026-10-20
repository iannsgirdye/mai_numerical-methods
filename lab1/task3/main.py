"""Задание 1.3. Метод простых итераций (Якоби) и метод Зейделя."""

from pathlib import Path

import numpy as np

# Настройки отображения результатов и максимальное число итераций.
ARRAY_DECIMAL_PLACES = 8
ARRAY_LINE_WIDTH = 120
MAX_ITERATIONS = 100_000
COMPARISON_SCIENTIFIC_DIGITS = 3

np.set_printoptions(precision=ARRAY_DECIMAL_PLACES, suppress=False, linewidth=ARRAY_LINE_WIDTH)


def read_input():
    with Path(__file__).with_name("input.txt").open(encoding="utf-8") as file:
        return [[float(value) for value in line.split()] for line in file if line.strip()]


def max_abs_value(values):
    """Вернуть наибольший по модулю элемент массива (например, невязки)."""
    values = np.asarray(values)
    return float(np.max(np.abs(values))) if values.size else 0.0


def jacobi(A, b, eps):
    """Решить A · x = b методом простых итераций (методом Якоби).

    При вычислении нового значения используются только значения x
    с предыдущей итерации — весь новый вектор считается отдельно.
    """
    n = len(b)
    x = np.zeros(n)  # Начальное приближение: все неизвестные равны нулю.

    for iteration in range(1, MAX_ITERATIONS + 1):
        new_x = np.zeros(n)
        for i in range(n):
            # Из i-го уравнения выражаем x_i. Остальные значения берём из старого x.
            sum_without_diagonal = A[i] @ x - A[i, i] * x[i]
            new_x[i] = (b[i] - sum_without_diagonal) / A[i, i]
        if max_abs_value(new_x - x) < eps:
            return new_x, iteration
        x = new_x

    raise RuntimeError(f"Метод Якоби не сошёлся за {MAX_ITERATIONS} итераций. Проверьте систему и точность.")


def seidel(A, b, eps):
    """Решить A · x = b методом Зейделя."""
    n = len(b)
    x = np.zeros(n)  # Начальное приближение.

    for iteration in range(1, MAX_ITERATIONS + 1):
        old_x = x.copy()
        for i in range(n):
            left_sum = A[i, :i] @ x[:i]
            right_sum = A[i, i + 1:] @ old_x[i + 1:]
            x[i] = (b[i] - left_sum - right_sum) / A[i, i]
        if max_abs_value(x - old_x) < eps:
            return x, iteration

    raise RuntimeError(f"Метод Зейделя не сошёлся за {MAX_ITERATIONS} итераций. Проверьте систему и точность.")


def main():
    data = read_input()
    n = int(data[0][0])

    # Следующие n строк — расширенная матрица: коэффициенты и правая часть.
    Augmented = np.array(data[1:n + 1], dtype=float)
    A = Augmented[:, :n]
    b = Augmented[:, n]
    eps = data[n + 1][0]

    # Решаем систему двумя разными итерационными методами.
    x_jacobi, iterations_jacobi = jacobi(A, b, eps)
    x_seidel, iterations_seidel = seidel(A, b, eps)

    # NumPy вычисляет эталонное решение для текущих A и b.
    reference = np.linalg.solve(A, b)
    residual_jacobi = max_abs_value(A @ x_jacobi - b)
    residual_seidel = max_abs_value(A @ x_seidel - b)

    print(f"Заданная точность ε:\n{eps:g}\n")
    print("Метод простых итераций (Якоби)\n")
    print(f"Решение x:\n{x_jacobi}\n")
    print(f"Количество итераций:\n{iterations_jacobi}\n")
    print(f"Максимальная невязка |A · x − b|:\n{residual_jacobi:.{COMPARISON_SCIENTIFIC_DIGITS}e}\n")
    print(f"Отклонение от эталонного решения:\n{max_abs_value(x_jacobi - reference):.{COMPARISON_SCIENTIFIC_DIGITS}e}\n")
    print("Метод Зейделя\n")
    print(f"Решение x:\n{x_seidel}\n")
    print(f"Количество итераций:\n{iterations_seidel}\n")
    print(f"Максимальная невязка |A · x − b|:\n{residual_seidel:.{COMPARISON_SCIENTIFIC_DIGITS}e}\n")
    print(f"Отклонение от эталонного решения:\n{max_abs_value(x_seidel - reference):.{COMPARISON_SCIENTIFIC_DIGITS}e}\n")
    print(f"Эталонное решение (NumPy):\n{reference}\n")

    if iterations_jacobi < iterations_seidel:
        faster = "метод простых итераций (Якоби)"
    elif iterations_seidel < iterations_jacobi:
        faster = "метод Зейделя"
    else:
        faster = "оба метода потребовали одинаковое число итераций"
    print(f"Сравнение по числу итераций:\nБыстрее сходится {faster}.\n")


if __name__ == "__main__":
    main()
