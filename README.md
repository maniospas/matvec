# matvec

A domain-specific language for fast graph shift operations.
This implements mathematical fields on numbers,
n-dimensional column vectors, and n-by-n sparse matrices.

**License:** Apache Software License
<br>**Author:** Emmanouil (Manios) Krasanakis
<br>**Dependncies:** *numpy*

# :zap: Quickstart
Creating a 5-dimensional vector (can use `numpy` arrays 
as inputs interchangeably with lists everywhere):
```python
from matvec import Vector
x = Vector([1, 2, 3, 4, 5])
```

Creating a 5x5 sparse matrix A in coo-format 
with non-zero elements A[1,2]=9 and A[3,0]=21
```python
from matvec import Matrix
A = Matrix([1, 2],
           [3, 0],
           [9, 21],
           5)
```

Print the outcome of matrix-vector multiplication:
```python
print(A*x)
```

Print the outcome of left-multiplying transpose(x)
with A:
```python
print(x*A)
```

# :fire: Features
:rocket: Parallelized matrix-vector multiplication.<br>
:chart_with_downwards_trend: Memory reuse optimization.<br>
:mag: numpy compatibility.<br>
:factory: Common arithmetic operations.<br>

# :volcano: Benchmark
Benchmarks tested on a machine with 2.6 GHz CPU base clock and
up to 4.4 GHz turbo boost, 12 logical
cores, and 16GB DDR3 RAM. They span vectors of 1.E5 to
1.E6 elements and matrices with 20x the number of
non-zeroes.
More rigorous evaluation will take place in the future.


| Task                                      | numpy/scipy | matvec    |
|-------------------------------------------|-------------|-----------|
| Create new vector or array                | 0.026 sec   | 0.015 sec |
| 1000 temp. additions of 1.E6 vectors only | 2.130 sec   | 1.061 sec |
| Create matrix                             | 1.049 sec   | 0.378 sec |
| Sparse matrix with vector multiplication  | 0.090 sec   | 0.021 sec |

![benchmarks](benchmarks.png)


# :memo: List of Operations
* Full arithmetic operations `* + - / == < > <= >=` between
vectors and other vectors or scalars.
* Matrix-vector multiplication `*` (both left and right).
* Element access and assignment for vectors with `[]`.
* Masking, such as `y = x[x>0]`.
* `matvec.clear()` Clears cache.