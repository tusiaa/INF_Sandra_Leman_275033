import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.backend.topology import Pyramid
from pyswarms.backend.topology import Ring
from pyswarms.backend.topology import Random

# Run optimizer
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
my_topology = Pyramid(static=False)
optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                    options=options, topology=my_topology)

# historia kosztów i pozycji
cost_history, pos_history = optimizer.optimize(fx.sphere, iters=50)

#tworzenie animacji
m = Mesher(func=fx.sphere)
animation = plot_contour(pos_history=pos_history,
 mesher=m,
mark=(0, 0))
animation.save('plot01.gif', writer='imagemagick', fps=10)


# Run optimizer
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
my_topology = Ring(static=False)
optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                    options=options, topology=my_topology)

# historia kosztów i pozycji
cost_history, pos_history = optimizer.optimize(fx.sphere, iters=50)

#tworzenie animacji
m = Mesher(func=fx.sphere)
animation = plot_contour(pos_history=pos_history,
 mesher=m,
mark=(0, 0))
animation.save('plot02.gif', writer='imagemagick', fps=10)


# Run optimizer
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
my_topology = Random(static=False)
optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                    options=options, topology=my_topology)

# historia kosztów i pozycji
cost_history, pos_history = optimizer.optimize(fx.sphere, iters=50)

#tworzenie animacji
m = Mesher(func=fx.sphere)
animation = plot_contour(pos_history=pos_history,
 mesher=m,
mark=(0, 0))
animation.save('plot03.gif', writer='imagemagick', fps=10)