import math
import matplotlib.pyplot as plt

# Точность вычислений и начальное приближение
epsilon = 1e-8
x0 = 6.0
y0 = 1.0
max_iterations = 1000

a = 4.0


# Исходная система уравнений
def f1(x, y):
    return (x ** 2 + a ** 2) * y - a ** 3


def f2(x, y):
    return (x - a / 2) ** 2 + (y - a / 2) ** 2 - a ** 2


# Функции для метода простой итерации
def phi1(x, y):
    return a / 2 + math.sqrt(a ** 2 - (y - a / 2) ** 2)


def phi2(x, y):
    return a ** 3 / (x ** 2 + a ** 2)


# Решение системы линейных уравнений 2x2 методом Крамера
def solve_linear_system(A, b):
    determinant = A[0][0] * A[1][1] - A[0][1] * A[1][0]

    if abs(determinant) < 1e-14:
        raise ArithmeticError("Определитель матрицы близок к нулю")

    dx = (b[0] * A[1][1] - A[0][1] * b[1]) / determinant
    dy = (A[0][0] * b[1] - b[0] * A[1][0]) / determinant
    return dx, dy


# Норма вектора невязки
def residual_norm(x, y):
    return math.sqrt(f1(x, y) ** 2 + f2(x, y) ** 2)


# Метод простой итерации
def simple_iteration(x0, y0):
    x = x0
    y = y0
    history = []

    for k in range(1, max_iterations + 1):
        x_next = phi1(x, y)
        y_next = phi2(x, y)

        difference = math.sqrt((x_next - x) ** 2 + (y_next - y) ** 2)
        residual = residual_norm(x_next, y_next)
        history.append((k, x_next, y_next, difference, residual))

        if difference < epsilon and residual < epsilon:
            return x_next, y_next, history

        x = x_next
        y = y_next

    raise RuntimeError("Метод простой итерации не достиг заданной точности")


# Метод Ньютона
def newton_method(x0, y0):
    x = x0
    y = y0
    history = []

    for k in range(1, max_iterations + 1):
        # Матрица Якоби
        J = [
            [2 * x * y, x ** 2 + a ** 2],
            [2 * (x - a / 2), 2 * (y - a / 2)]
        ]

        # Решаем J * delta = -F
        b = [-f1(x, y), -f2(x, y)]
        dx, dy = solve_linear_system(J, b)

        x_next = x + dx
        y_next = y + dy

        difference = math.sqrt(dx ** 2 + dy ** 2)
        residual = residual_norm(x_next, y_next)
        history.append((k, x_next, y_next, difference, residual))

        if difference < epsilon and residual < epsilon:
            return x_next, y_next, history

        x = x_next
        y = y_next

    raise RuntimeError("Метод Ньютона не достиг заданной точности")


# Графическое исследование системы: строим линии f1(x, y) = 0 и f2(x, y) = 0
x_values = [i / 50 for i in range(0, 401)]
y_values = [i / 50 for i in range(0, 201)]
F1 = [[f1(x, y) for x in x_values] for y in y_values]
F2 = [[f2(x, y) for x in x_values] for y in y_values]

plt.figure(figsize=(8, 6))
plt.contour(x_values, y_values, F1, levels=[0], colors="blue")
plt.contour(x_values, y_values, F2, levels=[0], colors="orange")
plt.scatter([x0], [y0], color="red", marker="x", s=80, label="Начальное приближение")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Графическое решение системы")
plt.grid(True)
plt.legend(["f1(x, y) = 0", "f2(x, y) = 0", "Начальное приближение"])
plt.axis("equal")
plt.tight_layout()
plt.show()


# Вычисление решений двумя методами
simple_x, simple_y, simple_history = simple_iteration(x0, y0)
newton_x, newton_y, newton_history = newton_method(x0, y0)


# Вывод результатов
print(f"Точность: {epsilon}")
print(f"Начальное приближение: x = {x0}, y = {y0}\n")

for name, x, y, history in [
    ("Метод простой итерации", simple_x, simple_y, simple_history),
    ("Метод Ньютона", newton_x, newton_y, newton_history)
]:
    print(name)
    print(f"Решение: x = {x:.12f}, y = {y:.12f}")
    print(f"Количество итераций: {len(history)}")
    print("k | x(k) | y(k) | разность | невязка")
    for k, x_value, y_value, difference, residual in history:
        print(f"{k:2d} | {x_value:.10f} | {y_value:.10f} | {difference:.3e} | {residual:.3e}")
    print()

print("Сравнение методов")
print(f"Расстояние между решениями: {math.sqrt((simple_x - newton_x) ** 2 + (simple_y - newton_y) ** 2):.3e}")
print(f"Итераций простым методом: {len(simple_history)}")
print(f"Итераций методом Ньютона: {len(newton_history)}")
