import pyswarms as ps
import numpy
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

rozwiazanie = [[[3], [3], [4], [1], [1, 1]], [[2, 1], [3], [3], [1], [3]]]
# rozwiazanie = [[[3], [4], [4], [6], [4, 2], [8], [5, 2], [5], [1, 2], [5]],
#                [[5], [3, 1], [3, 1], [6], [7], [4], [6], [4, 2], [5], [2, 2]]]
# rozwiazanie = [[[1, 4, 3], [1, 3, 2], [1], [1, 1], [1, 1, 2], [9], [5, 3], [5], [1, 2], [1, 1, 2]],
#                [[8], [3, 1], [1, 4], [2, 5], [2, 1, 3], [2, 1], [2], [1, 3, 2], [2, 3, 2], [2]]]
# rozwiazanie = [[[3, 2], [1, 3, 2], [2, 1, 2, 1], [2, 1, 1], [2, 1], [1, 2, 1], [6, 6], [11],
#                 [6, 6], [1, 1, 1], [1, 1, 1], [1, 1], [1, 1, 2, 1], [1, 1, 2], [3]],
#                [[1, 1], [1, 1], [1, 1], [1, 1], [1, 3], [3, 3], [1, 2, 3, 3], [1, 1, 2, 1, 1, 1],
#                 [1, 5, 1, 1], [2, 3, 1], [1, 3, 1], [1, 1, 3, 1, 1], [2, 5, 1], [4, 1, 4]]]

width = len(rozwiazanie[0])
height = len(rozwiazanie[1])

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 10, 'p': 1}


# Sprawdza poprawność bloków
def fitness_func(solution):
    fitness = 0
    # Sprawdzanie kolumn
    for i in range(width):
        sum = 0
        place = 0
        isblock = False
        for j in range(height):
            if solution[j*width + i] and not isblock:
                block = 1
                isblock = True
            elif solution[j*width + i] and isblock:
                block = block + 1
            elif not solution[j*width + i] and isblock:
                isblock = False
                if place < len(rozwiazanie[0][i]):
                    sum = sum + numpy.abs(block - rozwiazanie[0][i][place])
                    place = place + 1
                else:
                    sum = sum + block
            if j == height - 1 and isblock:
                if place < len(rozwiazanie[0][i]):
                    sum = sum + numpy.abs(block - rozwiazanie[0][i][place])
                else:
                    sum = sum + block
        fitness = fitness + sum

    # Sprawdzanie wierszy
    for i in range(height):
        sum = 0
        place = 0
        isblock = False
        for j in range(width):
            if solution[i*width + j] and not isblock:
                block = 1
                isblock = True
            elif solution[i*width + j] and isblock:
                block = block + 1
            elif not solution[i*width + j] and isblock:
                isblock = False
                if place < len(rozwiazanie[1][i]):
                    sum = sum + numpy.abs(block - rozwiazanie[1][i][place])
                    place = place + 1
                else:
                    sum = sum + block
            if j == width - 1 and isblock:
                if place < len(rozwiazanie[1][i]):
                    sum = sum + numpy.abs(block - rozwiazanie[1][i][place])
                else:
                    sum = sum + block
        fitness = fitness + sum

    return fitness*fitness


def f(x):
    n_particles = x.shape[0]
    j = [fitness_func(x[i]) for i in range(n_particles)]
    return numpy.array(j)


optimizer = ps.discrete.BinaryPSO(n_particles=100, dimensions=width*height, options=options)
cost, pos = optimizer.optimize(f, iters=500, verbose=True)

print("Parameters of the best solution : {solution}".format(solution=pos))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=cost))
print("Predicted output based on the best solution :")
for i in range(height):
    for j in range(width):
        if pos[i*width+j]:
            print('X', end=' ')
        else:
            print(' ', end=' ')
    print("")

cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()
