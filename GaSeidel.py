import numpy as np

def GaussSeidel(A, b, es=1.e-7, maxit=50):
    """
    Implements the Gauss-Seidel method to solve a set of linear algebraic equations
    without relaxation.

    Input:
    A = coefficient matrix
    b = constant vector
    es = stopping criterion (default = 1.e-7)
    maxit = maximum number of iterations (default=50)

    Output:
    x = solution vector
    """
    n, m = np.shape(A)
    if n != m:
        return 'Coefficient matrix must be square'

    C = np.zeros((n, n))
    x = np.zeros((n, 1))

    # Set up C matrix with zeros on the diagonal
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = A[i, j]

    d = np.zeros((n, 1))

    # Divide C elements by A pivots
    for i in range(n):
        C[i, 0:n] = C[i, 0:n] / A[i, i]
        d[i] = b[i] / A[i, i]

    ea = np.zeros((n, 1))
    xold = np.zeros((n, 1))

    # Gauss-Seidel method
    for it in range(maxit):
        for i in range(n):
            xold[i] = x[i]  # Save the x's for convergence test

        for i in range(n):
            x[i] = d[i] - C[i, :].dot(x)  # Update the x's 1-by-1

            if x[i] != 0:
                ea[i] = abs((x[i] - xold[i]) / x[i])  # Compute change error

        if np.max(ea) < es:  # Exit for loop if stopping criterion met
            break

    if it == maxit:  # Check for maximum iteration exit
        return 'maximum iterations reached'
    else:
        return x

# Example matrix and vector
A = np.matrix('3. -0.1 -0.2 ; 0.1 7. -0.3 ; 0.3 -0.2 10.')
b = np.matrix('7.85 ; -19.3 ; 71.4')

x = GaussSeidel(A, b)
print('Solution is\n', x)

x2 = np.linalg.solve(A, b)
print('2nd solution is\n', x2)
