"""Задание 1.4. Метод вращений Якоби для собственных значений и векторов."""

from pathlib import Path

import numpy as np

# Настройки вывода и предел количества вращений.
ARRAY_DECIMAL_PLACES = 6
ARRAY_LINE_WIDTH = 120
MAX_ROTATIONS = 100_000
COMPARISON_SCIENTIFIC_DIGITS = 3

np.set_printoptions(precision=ARRAY_DECIMAL_PLACES, suppress=False, linewidth=ARRAY_LINE_WIDTH)


def read_input():
    with Path(__file__).with_name("input.txt").open(encoding="utf-8") as file:
        return [[float(value) for value in line.split()] for line in file if line.strip()]


def max_abs_difference(first, second):
    """Найти самое большое по модулю расхождение между двумя результатами."""
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def jacobi_eigen(A, eps):
    """Найти собственные значения и векторы симметричной матрицы вращениями Якоби.

    На каждом шаге выбирается наибольший по модулю элемент вне главной диагонали.
    Вращением этот элемент обнуляется; после многих шагов матрица становится почти диагональной.
    Столбцы V — собственные векторы.
    """
    n = len(A)
    A_work = A.copy().astype(float)
    V = np.eye(n)
    errors = []

    if n == 1:
        return np.array([A_work[0, 0]]), V, [0.0]

    for step in range(MAX_ROTATIONS):
        # Находим индексы p, q наибольшего внедиагонального элемента.
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A_work[i, j]) > abs(A_work[p, q]):
                    p, q = i, j

        error = abs(A_work[p, q])
        errors.append(error)
        if error < eps:
            break

        # Рассчитываем угол поворота так, чтобы новый элемент A_work[p, q] стал нулём.
        apq = A_work[p, q]
        tau = (A_work[q, q] - A_work[p, p]) / (2 * apq)
        t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau * tau)) if tau != 0 else 1.0
        c = 1 / np.sqrt(1 + t * t)
        s = t * c
        app, aqq = A_work[p, p], A_work[q, q]

        # Обновляем строки и столбцы p и q, сохраняя симметричность матрицы.
        for k in range(n):
            if k != p and k != q:
                x, y = A_work[k, p], A_work[k, q]
                A_work[k, p] = A_work[p, k] = c * x - s * y
                A_work[k, q] = A_work[q, k] = s * x + c * y

        # Обновляем два диагональных элемента и обнуляем выбранную пару.
        A_work[p, p] = app - t * apq
        A_work[q, q] = aqq + t * apq
        A_work[p, q] = A_work[q, p] = 0

        # Те же вращения применяем к столбцам матрицы собственных векторов V.
        V[:, [p, q]] = np.column_stack((c * V[:, p] - s * V[:, q], s * V[:, p] + c * V[:, q]))
    else:
        raise RuntimeError(f"Метод вращений Якоби не сошёлся за {MAX_ROTATIONS} шагов.")

    lambda_values = np.diag(A_work).copy()
    order = np.argsort(lambda_values)
    return lambda_values[order], V[:, order], errors


def main():
    data = read_input()
    n = int(data[0][0])

    A = np.array(data[1:n + 1], dtype=float)
    eps = data[n + 1][0]

    lambda_values, V, errors = jacobi_eigen(A, eps)
    Lambda = np.diag(lambda_values)
    A_V = A @ V
    V_Lambda = V @ Lambda

    print(f"Заданная точность ε:\n{eps:g}\n")
    print(f"Количество вращений:\n{len(errors) - 1}\n")
    print(f"Собственные значения λ:\n{lambda_values}\n")
    print(f"Матрица собственных векторов V (векторы расположены по столбцам):\n{V}\n")
    print(f"Диагональная матрица Λ:\n{Lambda}\n")
    print(f"Произведение A · V:\n{A_V}\n")
    print(f"Произведение V · Λ:\n{V_Lambda}\n")
    print(f"Проверка A · V = V · Λ (максимальное расхождение):\n{max_abs_difference(A_V, V_Lambda):.{COMPARISON_SCIENTIFIC_DIGITS}e}\n")
    print("Максимальный внедиагональный элемент по шагам:")
    for step, error in enumerate(errors):
        print(f"Шаг {step}: {error:.{COMPARISON_SCIENTIFIC_DIGITS}e}")
    print()


if __name__ == "__main__":
    main()
