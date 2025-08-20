def newton(x0, f):
    """Returns the numerical approximation of a function using Newton Raphson."""
    if not callable(f): #check for valid arguments
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")
    eps = 1e-5
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
