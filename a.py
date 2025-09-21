import numpy as np

a = np.array([1, 2, 3, 4])
print("Исходный массив:", a)

# Вставим 99 на позицию с индексом 2
b = np.insert(a, len(a), 99)

print("Новый массив:", b)