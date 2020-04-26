import sigpy as sp
import numpy as np
import matplotlib.pyplot as plt
import eboxFilter as ebox


sigma = 2.0
k = 1.0
r, alpha, c1, c2 = ebox.preComputeConstants(sigma, k)
size = 600
N = 7
a1 = np.ones(size)
a1 = np.random.rand(size)
a2 = sp.to_device(ebox.eboxFilter(a1, r, c1, c2, device=-1))
a3 = np.convolve(a1, np.ones((N,)) / N, mode="same")
# plt.plot(np.arange(size), a1, label="Original")
plt.plot(np.arange(size), a2, label="Running")
plt.plot(np.arange(size), a3, label="Convolve")
plt.legend()
plt.show()
