import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
x_max = [2, 2]
x_min = [1, 1]
my_bounds = (x_min, x_max)
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=my_bounds)
optimizer.optimize(fx.sphere, iters=1000)
