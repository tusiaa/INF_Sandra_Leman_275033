import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

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
cost_history, pos_history = optimizer.optimize(fx.easom, iters=50)

#tworzenie animacji
m = Mesher(func=fx.easom)
animation = plot_contour(pos_history=pos_history,
 mesher=m,
mark=(0, 0))
animation.save('plot1.gif', writer='imagemagick', fps=10)


# historia kosztów i pozycji
cost_history, pos_history = optimizer.optimize(fx.levi, iters=50)

#tworzenie animacji
m = Mesher(func=fx.levi)
animation = plot_contour(pos_history=pos_history,
 mesher=m,
mark=(0, 0))
animation.save('plot2.gif', writer='imagemagick', fps=10)



