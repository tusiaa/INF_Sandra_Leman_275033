import math
import pyswarms as ps
from matplotlib import pyplot as plt
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher, Designer

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Call instance of GlobalBestPSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                    options=options)


# Perform optimization
stats = optimizer.optimize(fx.easom, iters=100)

# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history

# Plot!
plot_cost_history(cost_history)
plt.show()


# Perform optimization
stats = optimizer.optimize(fx.levi, iters=100)

# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history

# Plot!
plot_cost_history(cost_history)
plt.show()


# historia kosztów i pozycji
stats = optimizer.optimize(fx.easom, iters=50)
cost_history = optimizer.cost_history
pos_history = optimizer.pos_history

#tworzenie animacji
d = Designer(limits=[(1, 5), (1, 5)])
m = Mesher(func=fx.easom, limits=[(-100, 100), (-100, 100)], delta=1)
animation = plot_contour(pos_history=pos_history, mesher=m, designer=d, mark=(math.pi, math.pi))
animation.save('plot1.gif', writer='imagemagick', fps=10)


# historia kosztów i pozycji
stats = optimizer.optimize(fx.levi, iters=50)
cost_history = optimizer.cost_history
pos_history = optimizer.pos_history

#tworzenie animacji
d = Designer(limits=[(-0.01, 1.2), (-4, 6)])
m = Mesher(func=fx.levi, limits=[(-10, 10), (-10, 10)], delta=1)
animation = plot_contour(pos_history=pos_history, mesher=m, designer=d, mark=(1, 1))
animation.save('plot2.gif', writer='imagemagick', fps=10)



