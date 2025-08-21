import numpy as np
from scipy.differentiate import jacobian

def newton(x0, f, eps=1e-6):
    """Returns the numerical approximation of a function using Newton Raphson."""
    if not callable(f): #check for valid arguments
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")
    diff = 1
    iter = 0
    while abs(diff) > eps:
        f1 = (f(x0 + eps) - f(x0 - eps)) / (eps * 2)
        f2 = (f(x0 + eps) - 2 * f(x0) + f(x0 - eps)) / (eps**2)
        if f2 == 0: #check for zero division
            raise RuntimeError(f"Second derivative zero, choose different start value.")
        diff = f1 / f2
        x0 = x0 - diff
        iter += 1
        print(x0, diff)
        if x0 > 1e7: #check for divergence
            raise RuntimeError(f"At iteration {iter}, optimization appears to be diverging")
    return x0

def multi_newton(x0, f, eps=1e-6):
    """Returns the numerical approximation of a multivariate function using Newton Raphson."""
    if not callable(f): #check for valid arguments
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")
    diff = 1
    max_iter = 200
    for i in range(max_iter):
        # Check for convergence
        if diff < eps:
            print(f"Converged in {i+1} iterations.")
            return x0
        try:
            # Solve the linear system J(x) * delta_x = -F(x) for delta_x
            delta_x = np.linalg.solve(jacobian(f, x0), -f(x0))
            x0 = x0 + delta_x
        except np.linalg.LinAlgError:
            print("Jacobian matrix is singular, cannot proceed.")
            return None
        iter += 1
        diff = np.linalg.norm(delta_x)
        print(x0, diff)
    print("Maximum iterations reached, no convergence.")
    return None
