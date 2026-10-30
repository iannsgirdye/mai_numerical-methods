import math
import matplotlib.pyplot as plt

# Точность вычислений и начальное приближение
epsilon = 1e-8
x0 = 0.1
max_iterations = 1000


# Исходное уравнение: sqrt(1 - x^2) - e^x + 0.1 = 0
def f(x):
    return math.sqrt(1 - x ** 2) - math.exp(x) + 0.1


# Функция для метода простой итерации: x = phi(x)
def phi(x):
    return math.log(math.sqrt(1 - x ** 2) + 0.1)


# Производная исходной функции
def df(x):
    return -x / math.sqrt(1 - x ** 2) - math.exp(x)


# Метод простой итерации
def simple_iteration(x0):
    x = x0
    history = []

    for k in range(1, max_iterations + 1):
        x_next = phi(x)
        difference = abs(x_next - x)
        residual = abs(f(x_next))
        error = abs(f(x_next) / df(x_next))
        history.append((k, x_next, difference, residual, error))

        if difference < epsilon and residual < epsilon:
            return x_next, history

        x = x_next

    raise RuntimeError("Метод простой итерации не достиг заданной точности")


# Метод Ньютона
def newton_method(x0):
    x = x0
    history = []

    for k in range(1, max_iterations + 1):
        x_next = x - f(x) / df(x)
        difference = abs(x_next - x)
        residual = abs(f(x_next))
        error = abs(f(x_next) / df(x_next))
        history.append((k, x_next, difference, residual, error))

        if difference < epsilon and residual < epsilon:
            return x_next, history

        x = x_next

    raise RuntimeError("Метод Ньютона не достиг заданной точности")


# Графическое исследование уравнения
x_values = [i / 1000 for i in range(0, 991)]
y1_values = [math.sqrt(1 - x ** 2) for x in x_values]
y2_values = [math.exp(x) - 0.1 for x in x_values]

plt.figure(figsize=(9, 5))
plt.plot(x_values, y1_values, label="sqrt(1 - x^2)")
plt.plot(x_values, y2_values, label="exp(x) - 0.1")
plt.scatter([x0], [math.sqrt(1 - x0 ** 2)], color="red", label="Начальное приближение")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Графическое решение уравнения")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Вычисление корней двумя методами
simple_root, simple_history = simple_iteration(x0)
newton_root, newton_history = newton_method(x0)


# Вывод результатов
print(f"Точность: {epsilon}")
print(f"Начальное приближение: {x0}\n")

for name, root, history in [
    ("Метод простой итерации", simple_root, simple_history),
    ("Метод Ньютона", newton_root, newton_history)
]:
    print(name)
    print(f"Корень: {root:.12f}")
    print(f"Количество итераций: {len(history)}")
    print("k | x(k) | разность | невязка | оценка ошибки")
    for k, x, difference, residual, error in history:
        print(f"{k:2d} | {x:.12f} | {difference:.3e} | {residual:.3e} | {error:.3e}")
    print()

print("Сравнение методов")
print(f"Разность корней: {abs(simple_root - newton_root):.3e}")
print(f"Итераций простым методом: {len(simple_history)}")
print(f"Итераций методом Ньютона: {len(newton_history)}")
