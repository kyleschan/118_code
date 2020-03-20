import numpy as np

# Initialize matrix A
A = np.array([[0.18, 0.51333, 0.18, 0.18, 0.18],
              [0.51333, 0.18, 0.68, 0.51333, 0.68],
              [0.51333, 0.18, 0.18, 0.51333, 0.18],
              [0.18, 0.51333, 0.68, 0.18, 0.68],
              [0.51333, 0.51333, 0.18, 0.51333, 0.18]])

# Returns the approximated eigenvalue
def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

# The main algorithm which takes the matrix A
# and the number of iterations as arguments
def power_method(A, iterations):
    n, d = A.shape

# Randomly initialize v with the uniform distribution
# over the unit interval
    v = np.random.random(d)
    ev = eigenvalue(A, v)

    for i in range(iterations):
        # Apply matrix A to v
        Av = A.dot(v)

        # Normalize v to have unit norm since it's a probability distribution
        v_new = Av / np.linalg.norm(Av)

        # Get new approximation for leading eigenvalue
        ev_new = eigenvalue(A, v_new)

        # Update the approximations
        v = v_new
        ev = ev_new

        # Print iteration number and approximations
        print("Iteration " + str(i + 1))
        print(v)
        print(ev)

    return v, ev


power_method(A, 1000)
