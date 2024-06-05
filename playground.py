import matvec


# A = [1, 1, 1],
#     [0, 2, 0]
#     [0, 0, 0])

x = matvec.Vector([1, 2, 3])
A = matvec.Matrix([0, 0, 0, 1],
                  [0, 1, 2, 1],
                  [1, 1, 1, 2],
                  len(x))

print(matvec.sum(A, axis=0))


print(matvec.multiply(A, x))