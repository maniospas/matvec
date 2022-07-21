import random
import numpy as np
import matvec
import scipy.sparse as sp
from timeit import default_timer as time
from tqdm import tqdm

matvec.load_matvec("build/lib.win-amd64-3.10/matvec/matvec/py.pyd")


class Timer:
    def __init__(self):
        self.values = list()
        self._tic = None

    def __enter__(self):
        self._tic = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        matvec.clear()  # cold-start performance
        self.values.append(time()-self._tic)

    def mean(self):
        return np.mean(self.values)

numpy_vector_allocation = Timer()
scipy_matrix_allocation = Timer()
scipy_multiply = Timer()
matvec_vector_allocation = Timer()
matvec_matrix_allocation = Timer()
matvec_multiply = Timer()
datasize = list()

for iter in tqdm(range(100)):
    volume = 10000*(iter//10+1)
    datasize.append(volume)
    density = 20
    x = np.random.choice(list(range(volume)), volume*density, replace=True)
    y = np.random.choice(list(range(volume)), volume*density, replace=True)
    values = [random.random() for _ in range(volume*density)]
    vector = [random.random() for _ in range(volume)]

    with numpy_vector_allocation:
        scipy_vector = np.array(vector)
    with scipy_matrix_allocation:
        scipy_matrix = sp.coo_matrix((values, (x, y)), shape=(len(vector), len(vector)))
    with matvec_vector_allocation:
        matvec_vector = matvec.Vector(vector)
    with matvec_matrix_allocation:
        matvec_matrix = matvec.Matrix(x, y, values, len(vector))
    with scipy_multiply:
        scipy_result = scipy_matrix * scipy_vector
    with matvec_multiply:
        matvec_result = matvec_matrix * matvec_vector

print(f"Measure\t\t numpy/scipy\t matvec")
print(f"Vector allocation\t {numpy_vector_allocation.mean():.3f} sec\t {matvec_vector_allocation.mean():.3f} sec")
print(f"Matrix allocation\t {scipy_matrix_allocation.mean():.3f} sec\t {matvec_matrix_allocation.mean():.3f} sec")
print(f"Matrix-vector multiplication\t {scipy_multiply.mean():.3f} sec\t {matvec_multiply.mean():.3f} sec")

from matplotlib import pyplot as plt
plt.subplot(1, 3, 1)
plt.plot(datasize, numpy_vector_allocation.values, label="numpy")
plt.plot(datasize, matvec_vector_allocation.values, label="matvec")
plt.ylabel("sec")
plt.xlabel("Vector length")
plt.legend()
plt.title("Vector allocation")
plt.subplot(1, 3, 2)
plt.plot(datasize, scipy_matrix_allocation.values, label="scipy")
plt.plot(datasize, matvec_matrix_allocation.values, label="matvec")
plt.ylabel("sec")
plt.xlabel("Vector length")
plt.legend()
plt.title("Matrix allocation")
plt.subplot(1, 3, 3)
plt.plot([v*density for v in datasize], scipy_multiply.values, label="scipy")
plt.plot([v*density for v in datasize], matvec_multiply.values, label="matvec")
plt.ylabel("sec")
plt.xlabel("Non-zeroes")
plt.legend()
plt.title("Matrix-vector multiplication")
plt.show()
