import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.stats as stats
from matplotlib import rc

lim_min = -60
lim_max = 90
N_SAMPLES = 1000

x = np.linspace(lim_min+(0.1*lim_min), lim_max+(0.1*lim_max), N_SAMPLES)
y1 = stats.norm.pdf(x[x < 0], 0, 0.5*(-lim_min))
y2 = stats.norm.pdf(x[x >= 0], 0, 0.5*(lim_max))
y = np.concatenate((y1, y2))

rc('axes', linewidth=2)
rc('grid', linewidth=2)
rc('font', weight='bold', size=10)

frame = plt.gca()
plt.plot(x, y, color='b', linewidth=3.0)
plt.plot(x, stats.uniform.pdf(x, lim_min, lim_max-lim_min), color='r', linewidth=3.0)

plt.axvline(x=0, color='k', linestyle='--', linewidth=2.0, alpha=0.5)
plt.axvline(x=lim_min, color='g', linestyle='--', linewidth=2.0, alpha=0.5)
plt.axvline(x=lim_max, color='g', linestyle='--', linewidth=2.0, alpha=0.5)


plt.legend(['baseline distribution', 'uniform distribution', "mean", "limits"], fontsize=14)
plt.xlabel('Angle (degrees)', fontsize=14, fontweight='bold')

frame.axes.get_yaxis().set_visible(False)

# x = np.linspace(0, lim_max, N_SAMPLES)
# plt.plot(x, stats.norm.pdf(x, 0, 0.5*(lim_max)), color='b')

# plt.grid(True)


# Show the plot
# sns.displot(r_u)

# fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# ax[0].hist(r_n, bins=1000, range=(lim_min, lim_max), color='b', alpha=1.0)
# ax[0].hist(r_u, bins=1000, range=(lim_min, lim_max), color='r', alpha=0.5)

plt.show()