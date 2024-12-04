from timeit import default_timer as time

import matvec
import numpy as np

x = np.random.random(1000000)
mx = matvec.Vector(x)


tic = time()
for i in range(1000):
    mx = mx+mx
toc = time()
print(f"Matvec {toc-tic:.3f}")


tic = time()
for i in range(1000):
    x = x+x
toc = time()
print(f"NUmpy {toc-tic:.3f}")
