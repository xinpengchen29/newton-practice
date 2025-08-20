def newton(x0, f):
    eps = 1e-5
    diff = 1
    while abs(diff) > eps:
        f1 = (f(x0 + eps) - f(x0 - eps)) / (eps*2)
        f2 = (f(x0 + eps) - 2 * f(x0) + f(x0 - eps)) / (eps**2)
        diff = f1/f2
        x0 = x0 - diff
        print(x0, diff)
    return x0
