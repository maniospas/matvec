# matvec

A domain-specific language for fast graph shift operations.
This implements mathematical fields on numbers,
n-dimensional column vectors, and n-by-n sparse matrices.

**License:** Apache Software License
<br>**Author:** Emmanouil (Manios) Krasanakis
<br>**Dependncies:** ---

# :zap: Quickstart
Creating a 5-dimensional vector:
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
:rocket: more than twice faster than scipy for matrix-vector multiplication.<br>
:mag: TODO: numpy compatibility.<br>
:factory: Common arithmetic operations.<br>