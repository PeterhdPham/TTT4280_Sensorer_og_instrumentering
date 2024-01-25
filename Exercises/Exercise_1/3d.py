import numpy as np
from numpy.random import default_rng # anbefales å bruke dette
delta_r = 100
rng = default_rng()
r_values = 10000 + 2*delta_r * (rng.random(100000) - 0.5)
relative_std = np.std(r_values)/np.mean(r_values)
print(relative_std)
print(r_values)

delta_r2 = 50
r_values2= 5000 + 2*delta_r2 * (rng.random(100000) - 0.5)
r_values3= 5000 + 2*delta_r2 * (rng.random(100000) - 0.5)

standard_deviation_12=np.sqrt(np.var(r_values2)+np.var(r_values3))
print(standard_deviation_12)

mean_r_values23 = np.mean(np.concatenate([r_values2, r_values3]))

relative_std_combined = (standard_deviation_12 / mean_r_values23) * 100
print(relative_std_combined)