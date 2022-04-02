import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.plotters import plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Designer

# Run optimizer
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

# historia kosztów i pozycji
stats = optimizer.optimize(fx.sphere, iters=50)

#tworzenie animacji
m = Mesher(func=fx.sphere)
d = Designer(limits=[(-1, 1), (-1, 1), (-0.1, 1)])
pos_history = m.compute_history_3d(optimizer.pos_history)
animation = plot_surface(pos_history=pos_history, mesher=m, designer=d, mark=(0, 0, 0))
animation.save('plot00.gif', writer='imagemagick', fps=10)

