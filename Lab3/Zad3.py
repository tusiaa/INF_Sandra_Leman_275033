import pygad
import numpy
import time

lab = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
       [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
       [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
       [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
       [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
       [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
       [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
       [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
       [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [1, 2, 3, 4]

#definiujemy funkcjÄ fitness
def fitness_func(solution, solution_idx):
    x = 1
    y = 1
    for i in range(30):
        match solution[i]:
            case 1:
                if lab[x+1][y] == 0:
                    x += 1
                elif lab[x+1][y] == 3:
                    x += 1
                    for j in range(i+1, 30):
                        solution[j] = 0
                elif lab[x+1][y] == 1:
                    solution[i] = 0
            case 2:
                if lab[x-1][y] == 0:
                    x -= 1
                elif lab[x-1][y] == 3:
                    x -= 1
                    for j in range(i+1, 30):
                        solution[j] = 0
                elif lab[x-1][y] == 1:
                    solution[i] = 0
            case 3:
                if lab[x][y-1] == 0:
                    y -= 1
                elif lab[x][y-1] == 3:
                    y -= 1
                    for j in range(i+1, 30):
                        solution[j] = 0
                elif lab[x][y-1] == 1:
                    solution[i] = 0
            case 4:
                if lab[x][y+1] == 0:
                    y += 1
                elif lab[x][y+1] == 3:
                    y += 1
                    for j in range(i+1, 30):
                        solution[j] = 0
                elif lab[x][y+1] == 1:
                    solution[i] = 0
    fitness = -(10 - x + 10 - y)
    return fitness

fitness_function = fitness_func

#ile chromsomĂłw w populacji
#ile genow ma chromosom
sol_per_pop = 200
num_genes = 30

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 100
num_generations = 100
keep_parents = 5

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 4

t = 0
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

    #uruchomienie algorytmu
    start = time.time()
    ga_instance.run()
    end = time.time()
    print(end - start)
    t = t + (end - start)

print('Średnia')
print(t / 10)

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = []
for i in solution:
    match i:
        case 1:
            prediction.append("D")
        case 2:
            prediction.append("G")
        case 3:
            prediction.append("L")
        case 4:
            prediction.append("P")
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()
