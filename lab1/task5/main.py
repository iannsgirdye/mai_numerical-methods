"""Задание 1.5. QR-разложение и QR-алгоритм для собственных значений."""

from pathlib import Path

import numpy as np

# Настройки вывода и параметры QR-алгоритма.
ARRAY_DECIMAL_PLACES = 6
ARRAY_LINE_WIDTH = 120
MAX_QR_ITERATIONS = 20_000
THRESHOLD_SCALE_FLOOR = 1.0
COMPARISON_SCIENTIFIC_DIGITS = 3

np.set_printoptions(precision=ARRAY_DECIMAL_PLACES, suppress=False, linewidth=ARRAY_LINE_WIDTH)


def read_input():
    """Прочитать числа из input.txt рядом с этим скриптом."""
    with Path(__file__).with_name("input.txt").open(encoding="utf-8") as file:
        return [[float(value) for value in line.split()] for line in file if line.strip()]


def read_example():
    """Прочитать отдельный пример матрицы из example.txt рядом со скриптом."""
    with Path(__file__).with_name("example.txt").open(encoding="utf-8") as file:
        data = [[float(value) for value in line.split()] for line in file if line.strip()]
    n = int(data[0][0])
    return np.array(data[1:n + 1], dtype=float)


def max_abs_difference(first, second):
    """Найти самое большое по модулю расхождение между двумя результатами."""
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def qr(A):
    """Выполнить QR-разложение A = Q · R отражениями Хаусхолдера.

    Q — ортогональная матрица; R — верхняя треугольная матрица.
    Готовая функция numpy.linalg.qr здесь не используется: разложение строим сами.
    """
    n = len(A)
    R = A.copy().astype(float)  # Из копии A постепенно получим верхнюю треугольную R.
    Q = np.eye(n)               # В Q будем накапливать отражения Хаусхолдера.

    for k in range(n - 1):
        x = R[k:, k].copy()  # Часть столбца, которую нужно обнулить ниже диагонали.
        norm = np.sqrt(x @ x)
        if norm == 0:
            continue  # В этой части столбца уже точные нули; отражение не требуется.

        # Строим вектор отражения. Знак выбран так, чтобы избежать потери точности.
        sign = 1 if x[0] >= 0 else -1
        x[0] += sign * norm
        x /= np.sqrt(x @ x)  # Нормируем вектор отражения.

        # Применяем отражение к R справа, а соответствующее преобразование накапливаем в Q.
        R[k:, k:] -= 2 * np.outer(x, x @ R[k:, k:])
        Q[:, k:] -= 2 * np.outer(Q[:, k:] @ x, x)

    # После отражений Хаусхолдера R должна быть верхнетреугольной.
    # Обнуляем только нижнюю часть, где остались погрешности округления.
    R[np.tril_indices(n, k=-1)] = 0
    return Q, R


def eig2(a, b, c, d):
    """Найти собственные значения матрицы 2×2, включая комплексную пару при необходимости."""
    # Для матрицы [[a, b], [c, d]] используем формулу через след и дискриминант.
    discriminant = (a - d) ** 2 + 4 * b * c
    half_trace = (a + d) / 2
    if discriminant >= 0:
        root = np.sqrt(discriminant) / 2
        return [float(half_trace + root), float(half_trace - root)]

    # Отрицательный дискриминант означает два комплексно-сопряжённых значения.
    root = np.sqrt(-discriminant) / 2
    return [complex(half_trace, root), complex(half_trace, -root)]


def qr_eigenvalues(A, eps):
    """Найти собственные значения матрицы повторными QR-итерациями со сдвигом.

    После сходимости матрица имеет квазитреугольный вид:
    на диагонали могут остаться блоки 1×1 и 2×2. Блок 2×2 обрабатываем отдельно.
    """
    n = len(A)
    H = A.copy().astype(float)
    # Один масштаб для относительных порогов позволяет учитывать величину коэффициентов A.
    matrix_scale = max_abs_difference(H, np.zeros_like(H))
    matrix_scale = max(matrix_scale, np.finfo(float).tiny)

    for iteration in range(MAX_QR_ITERATIONS):
        # Если элемент под диагональю стал очень маленьким, считаем его нулём.
        # Так матрица разделяется на независимые блоки.
        for i in range(1, n):
            threshold = eps * (abs(H[i - 1, i - 1]) + abs(H[i, i]) + matrix_scale * THRESHOLD_SCALE_FLOOR)
            if abs(H[i, i - 1]) < threshold:
                H[i, i - 1] = 0

        # Сдвиг берём из последнего диагонального элемента, чтобы ускорить сходимость.
        shift = H[-1, -1]
        Q, R = qr(H - shift * np.eye(n))
        H = R @ Q + shift * np.eye(n)

        # Проверяем, достаточно ли матрица близка к верхней квазитреугольной форме.
        deep_elements_are_small = all(abs(H[i, j]) <= eps * (abs(H[i, i]) + abs(H[j, j]) + matrix_scale * THRESHOLD_SCALE_FLOOR) for i in range(2, n) for j in range(i - 1))
        # Два соседних элемента под диагональю означали бы, что блоки ещё не отделились.
        subdiagonal_isolated = [abs(H[i, i - 1]) > eps * (abs(H[i - 1, i - 1]) + abs(H[i, i]) + matrix_scale * THRESHOLD_SCALE_FLOOR) for i in range(1, n)]
        adjacent_blocks_remain = any(subdiagonal_isolated[i] and subdiagonal_isolated[i + 1] for i in range(len(subdiagonal_isolated) - 1))
        if deep_elements_are_small and not adjacent_blocks_remain:
            break
    else:
        raise RuntimeError(f"QR-алгоритм не сошёлся за {MAX_QR_ITERATIONS} итераций. Попробуйте другую точность.")

    # После сходимости элементы ниже первой поддиагонали должны быть нулевыми.
    H[np.tril_indices(n, k=-2)] = 0

    # Собираем собственные значения из блоков 1×1 и 2×2.
    values = []
    i = 0
    while i < n:
        if i + 1 < n and abs(H[i + 1, i]) > eps * (abs(H[i, i]) + abs(H[i + 1, i + 1]) + matrix_scale * THRESHOLD_SCALE_FLOOR):
            values.extend(eig2(H[i, i], H[i, i + 1], H[i + 1, i], H[i + 1, i + 1]))
            i += 2
        else:
            values.append(float(H[i, i]))
            i += 1

    return values, H, iteration + 1


def main():
    data = read_input()
    n = int(data[0][0])
    A = np.array(data[1:n + 1], dtype=float)
    eps = data[n + 1][0]

    # Сначала строим QR-разложение исходной матрицы и проверяем произведение Q · R.
    Q, R = qr(A)
    Q_R = Q @ R
    print(f"Исходная матрица A:\n{A}\n")
    print(f"Матрица Q:\n{Q}\n")
    print(f"Матрица R:\n{R}\n")
    print(f"Произведение Q · R:\n{Q_R}\n")
    print(f"Проверка A = Q · R (максимальное расхождение):\n{max_abs_difference(A, Q_R):.{COMPARISON_SCIENTIFIC_DIGITS}e}\n")

    # Теперь применяем QR-итерации для нахождения всех собственных значений.
    values, H_final, iterations = qr_eigenvalues(A, eps)
    print(f"Заданная точность ε:\n{eps:g}\n")
    print(f"Количество QR-итераций:\n{iterations}\n")
    print(f"Итоговая квазитреугольная матрица H:\n{H_final}\n")
    print(f"Собственные значения λ:\n{values}\n")

    # Отдельный пример читаем из example.txt, чтобы он не смешивался с основной задачей.
    # NumPy используется здесь для разрешённой условием проверки комплексной пары.
    A_example = read_example()
    example_values = np.linalg.eigvals(A_example)
    print(f"Пример матрицы из example.txt:\n{A_example}\n")
    print(f"Её собственные значения (проверка средствами NumPy):\n{example_values}\n")


if __name__ == "__main__":
    main()
