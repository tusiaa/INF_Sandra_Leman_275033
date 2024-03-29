import pygad
import numpy
import time

# rozwiazanie = [[[3], [3], [4], [1], [1, 1]], [[2, 1], [3], [3], [1], [3]]]
rozwiazanie = [[[3], [4], [4], [6], [4, 2], [8], [5, 2], [5], [1, 2], [5]],
               [[5], [3, 1], [3, 1], [6], [7], [4], [6], [4, 2], [5], [2, 2]]]
# rozwiazanie = [[[1, 4, 3], [1, 3, 2], [1], [1, 1], [1, 1, 2], [9], [5, 3], [5], [1, 2], [1, 1, 2]],
#                [[8], [3, 1], [1, 4], [2, 5], [2, 1, 3], [2, 1], [2], [1, 3, 2], [2, 3, 2], [2]]]
# rozwiazanie = [[[3, 2], [1, 3, 2], [2, 1, 2, 1], [2, 1, 1], [2, 1], [1, 2, 1], [6, 6], [11],
#                 [6, 6], [1, 1, 1], [1, 1, 1], [1, 1], [1, 1, 2, 1], [1, 1, 2], [3]],
#                [[1, 1], [1, 1], [1, 1], [1, 1], [1, 3], [3, 3], [1, 2, 3, 3], [1, 1, 2, 1, 1, 1],
#                 [1, 5, 1, 1], [2, 3, 1], [2, 3, 1], [1, 3, 1], [1, 1, 3, 1, 1], [2, 5, 1], [4, 1, 4]]]
# rozwiazanie = [[[5, 5], [3, 5, 3], [1, 1], [1, 2, 4, 3, 1], [1, 1, 6, 2, 1], [2, 1, 3, 3], [2, 1, 3, 3], [7, 1, 3],
#                 [2, 1, 3, 3], [2, 1, 3, 3], [1, 1, 6, 2, 1], [1, 2, 4, 3, 1], [1, 1], [3, 5, 3], [5, 5]],
#                [[5, 5], [3, 5, 3], [2, 9, 2], [1, 2, 1, 2, 1], [1, 1, 7, 1, 1], [4, 1, 4], [4, 1, 4], [13],
#                 [6, 6], [2, 7, 2], [1, 2, 2, 1], [1, 11, 1], [2, 9, 2], [3, 5, 3], [5, 5]]]

width = len(rozwiazanie[0])
height = len(rozwiazanie[1])

# definiujemy parametry chromosomu
# geny to liczby: 0 lub 1
gene_space = [0, 1]

# definiujemy funkcjÄ fitness
# Sprawdza poprawność bloków
def fitness_func(solution, solution_idx):
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
        fitness = fitness - sum

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
        fitness = fitness - sum

    return fitness

fitness_function = fitness_func

# ile chromsomĂłw w populacji
# ile genow ma chromosom
sol_per_pop = 200
num_genes = height * width

# ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
# ile pokolen
# ilu rodzicow zachowac (kilka procent)
num_parents_mating = 100
num_generations = 1000
keep_parents = 5

# jaki typ selekcji rodzicow?
# sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

# w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

# mutacja ma dzialac na ilu procent genow?
# trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = numpy.ceil(100 / num_genes)

t = 0
win = 0
best = -1000
best_sol = numpy.zeros(width*height)
for i in range(10):

# inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
    ga_instance = pygad.GA(gene_space=gene_space,
                           num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           stop_criteria=["reach_0"])

# uruchomienie algorytmu
    start = time.time()
    ga_instance.run()
    end = time.time()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(end - start)
    t = t + (end - start)
    if solution_fitness == 0:
        win = win + 1
    if solution_fitness > best:
        best = solution_fitness
        best_sol = solution

print('Średni czas')
print(t / 10)
print('Liczba poprawnych rozwiązań na 10 prób')
print(win)

# podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
print("Parameters of the best solution : {solution}".format(solution=best_sol))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best))

# tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
print("Predicted output based on the best solution :")
for i in range(height):
    for j in range(width):
        if best_sol[i*width+j]:
            print('X', end=' ')
        else:
            print(' ', end=' ')
    print("")

# wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen ostatniej próby
ga_instance.plot_fitness()
