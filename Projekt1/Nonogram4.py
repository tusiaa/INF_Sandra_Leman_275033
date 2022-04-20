import pygad
import numpy

# rozwiazanie = [[[3], [3], [4], [1], [1, 1]], [[2, 1], [3], [3], [1], [3]]]
rozwiazanie = [[[3], [4], [4], [6], [4, 2], [8], [5, 2], [5], [1, 2], [5]],
               [[5], [3, 1], [3, 1], [6], [7], [4], [6], [4, 2], [5], [2, 2]]]
# rozwiazanie = [[[1, 4, 3], [1, 3, 2], [1], [1, 1], [1, 1, 2], [9], [5, 3], [5], [1, 2], [1, 1, 2]],
#                [[8], [3, 1], [1, 4], [2, 5], [2, 1, 3], [2, 1], [2], [1, 3, 2], [2, 3, 2], [2]]]
# rozwiazanie = [[[3, 2], [1, 3, 2], [2, 1, 2, 1], [2, 1, 1], [2, 1], [1, 2, 1], [6, 6], [11],
#                 [6, 6], [1, 1, 1], [1, 1, 1], [1, 1], [1, 1, 2, 1], [1, 1, 2], [3]],
#                [[1, 1], [1, 1], [1, 1], [1, 1], [1, 3], [3, 3], [1, 2, 3, 3], [1, 1, 2, 1, 1, 1],
#                 [1, 5, 1, 1], [2, 3, 1], [1, 3, 1], [1, 1, 3, 1, 1], [2, 5, 1], [4, 1, 4]]]

width = len(rozwiazanie[0])
height = len(rozwiazanie[1])
column_blocks = []
for i in rozwiazanie[0]:
    for j in i:
        column_blocks.append(j)

# definiujemy parametry chromosomu
gene_space = range(width*height)

# def sort_func (n):
#     return n % width

# definiujemy funkcjÄ fitness
# Sprawdza poprawność bloków, liczby bloków i liczby zamalowanych kratek
def fitness_func(solution, solution_idx):
    fitness = 0
    solution2 = numpy.zeros(width*height)
    # solution.sort(key=sort_func)

    for i in range(len(solution)):
        for j in range(column_blocks[i]):
            if int(solution[i])+j*width < len(solution2):
                solution2[int(solution[i])+j*width] = 1

    # Sprawdzanie kolumn
    for i in range(width):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        place = 0
        blocks = 0
        isblock = False
        for j in range(len(rozwiazanie[0][i])):
            sum1 = sum1 + rozwiazanie[0][i][j]
        for j in range(height):
            sum2 = sum2 + solution2[j*width + i]
        for j in range(height):
            if solution2[j*width + i] and not isblock:
                blocks = blocks + 1
                block = 1
                isblock = True
            elif solution2[j*width + i] and isblock:
                block = block + 1
            elif not solution2[j*width + i] and isblock:
                isblock = False
                if place < len(rozwiazanie[0][i]):
                    sum3 = sum3 + numpy.abs(block - rozwiazanie[0][i][place])
                    place = place + 1
                else:
                    sum3 = sum3 + block
            if j == height - 1 and isblock:
                if place < len(rozwiazanie[0][i]):
                    sum3 = sum3 + numpy.abs(block - rozwiazanie[0][i][place])
                else:
                    sum3 = sum3 + block
        fitness = fitness - numpy.abs(sum1 - sum2) - numpy.abs(blocks - len(rozwiazanie[0][i])) - sum3

    # Sprawdzanie wierszy
    for i in range(height):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        place = 0
        blocks = 0
        isblock = False
        for j in range(len(rozwiazanie[1][i])):
            sum1 = sum1 + rozwiazanie[1][i][j]
        for j in range(width):
            sum2 = sum2 + solution2[i*width + j]
        for j in range(width):
            if solution2[i*width + j] and not isblock:
                blocks = blocks + 1
                block = 1
                isblock = True
            elif solution2[i*width + j] and isblock:
                block = block + 1
            elif not solution2[i*width + j] and isblock:
                isblock = False
                if place < len(rozwiazanie[1][i]):
                    sum3 = sum3 + numpy.abs(block - rozwiazanie[1][i][place])
                    place = place + 1
                else:
                    sum3 = sum3 + block
            if j == width - 1 and isblock:
                if place < len(rozwiazanie[1][i]):
                    sum3 = sum3 + numpy.abs(block - rozwiazanie[1][i][place])
                else:
                    sum3 = sum3 + block
        fitness = fitness - numpy.abs(sum1 - sum2) - numpy.abs(blocks - len(rozwiazanie[1][i])) - sum3

    return fitness

fitness_function = fitness_func

# ile chromsomĂłw w populacji
# ile genow ma chromosom
sol_per_pop = 200
num_genes = len(column_blocks)

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
ga_instance.run()

# podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
print("Predicted output based on the best solution :")
solution2 = numpy.zeros(width*height)

for i in range(len(solution)):
    for j in range(column_blocks[i]):
        if int(solution[i]) + j*width < len(solution2):
            solution2[int(solution[i]) + j*width] = 1

for i in range(height):
    for j in range(width):
        if solution2[i * width + j]:
            print('X', end=' ')
        else:
            print(' ', end=' ')
    print("")

# wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()
