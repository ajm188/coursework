from __future__ import print_function

import numpy as np
import numpy.linalg

# NUMBER 4, PART A
A = np.array(
    [
        [1, 0.5, 1.0/3.0],
        [0.5, 1.0/3.0, 0.25],
        [1.0/3.0, 0.25, 0.2],
    ],
)

for x in [0, 0.5, 1]:
    b = np.array([1, x, x ** 2])
    print(np.linalg.solve(A, b))

# NUMBER 4, PART C
A = np.array(
    [
        [9, -1.5, 3],
        [-1.5, 2.25, -1.5],
        [3, -1.5, 9],
    ],
)
b = np.array([2, -1, 0])
print(np.linalg.solve(A, b))
