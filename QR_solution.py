import numpy as np

#Set up the problem with the given numerical values and required arrays
b = np.array([1, 1, 5, 4])
R_1 = np.array([[-13.9642, -10.0972, -6.3018],
                [0, 6.9315, 3.0829],
                [0, 0, 3.7125]])
Q_transpose = np.transpose(np.array([[-0.0716, 0.9056, -0.3348, -0.2505],
              [-0.5729, 0.1753, 0.7675, -0.2282],
              [-0.6445, -0.3618, -0.5242, -0.4230],
              [-0.5013, 0.1354, -0.1552, 0.8404]]))

#Truncate b_1 to have same number of arguments as a row of R_1
b_1 = np.resize(np.dot(Q_transpose, b), R_1.shape[0])

#Print the solution to the linear system (R_1)x = b_1
print(np.linalg.solve(R_1, b_1))


