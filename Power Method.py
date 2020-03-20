import numpy as np

# Initialize matrix P_tilde
P_tilde = np.array([[0.18, 0.21333, 0.18, 0.18, 0.18],
              [0.21333, 0.18, 0.23, 0.21333, 0.23],
              [0.21333, 0.18, 0.18, 0.21333, 0.18],
              [0.18, 0.21333, 0.23, 0.18, 0.23],
              [0.21333, 0.21333, 0.18, 0.21333, 0.18]])


# The main algorithm which takes a matrix A
# and the number of iterations as arguments
def power_method(A, iterations):
    n, d = A.shape

    # Randomly initialize v with the uniform distribution
    # over the unit interval
    v = np.random.random(d)

    for i in range(iterations):
        # Apply matrix A to v
        Av = A.dot(v)

        # Normalize v to prevent numerical issues
        v_new = Av / np.linalg.norm(Av)

        # Print iteration number and approximation
        print("Iteration " + str(i + 1))
        print(v_new)

        # Update the approximation
        v = v_new

    return v


power_method(P_tilde, 1000)
