import matplotlib.pyplot as plt
import numpy as np


def f(x, a, b, c, func="cos"):
    if func == "cos":
        ftrig = np.cos
    elif func == "sin":
        ftrig = np.sin
    else:
        raise ValueError("Only cos/sin are supported")

    return np.sum(c[:, None] * ftrig(b[:, None] - a[:, None] * x[None, :]), axis=0)


def f_aux(x, x0, a, b, c, func="cos"):
    """
    Parameters
    ----------
    x: ndarray, shape (n_points)
        Input variable
    x0: ndarray, shape (1,)
    a: ndarray, shape (n_terms)
        Parameter 1
    b: ndarray, shape (n_terms)
        Parameter 1
    c: ndarray, shape (n_terms)
        Parameter 1
    """
    I = c > 0

    if func == "cos":
        e = b - a * x0  # (n_terms, n_points)

        # handle positive terms separately
        e[I] = np.pi - e[I]

        z = np.round(e / (2.0 * np.pi))
        zpi = 2 * z * np.pi
        phi = e - zpi

        y = b - zpi
        y[I] = b[I] + zpi[I] - np.pi

        """
        w = -0.5 * c * np.sinc(phi / np.pi)
        w[I] *= -1.0

        const = -c * (np.cos(phi) + 0.5 * phi * np.sin(phi))
        const[I] *= -1.0

        aux = np.sum(
            w[:, None] * (y[:, None] - a[:, None] * x[None, :]) ** 2 - const[:, None],
            axis=0,
        )
        """

    if func == "sin":

        e = 0.5 * np.pi - (b - a * x0)
        e[I] = np.pi - e[I]

        z = np.round(e / (2.0 * np.pi))
        zpi = 2 * z * np.pi
        phi = e - zpi

        y = b + zpi - 0.5 * np.pi
        y[I] = b[I] - zpi[I] + 0.5 * np.pi

    w = -0.5 * c * np.sinc(phi / np.pi)
    w[I] *= -1.0

    const = -c * (np.cos(phi) + 0.5 * phi * np.sin(phi))
    const[I] *= -1.0

    print(f"z: {z}, zpi: {zpi}, phi: {phi}, y: {y}, w:{w}")

    aux = np.sum(
        w[:, None] * (y[:, None] - a[:, None] * x[None, :]) ** 2 - const[:, None],
        axis=0,
    )

    return aux


if __name__ == "__main__":

    n_terms = 10
    x_lim = 8
    func = "cos"

    a = np.random.randn(n_terms)
    b = np.random.randn(n_terms)
    c = np.random.randn(n_terms)
    # a = np.array([0.33542565])
    # b = np.array([0.02449642])
    # c = np.array([-1.51912385])
    print("a", a)
    print("b", b)
    print("c", c)

    """
    a = np.array([1.0, -0.5])
    b = np.array([0.1, -0.3])
    c = -np.ones(n_terms)
    c = np.array([0.5, 0.3])
    """

    x = np.linspace(-x_lim * np.pi, x_lim * np.pi, 1000)
    x0 = x[int(0.8 * x.shape[0])]

    fx = f(x, a, b, c, func=func)
    fx_aux = f_aux(x, x0, a, b, c, func=func)

    gap = fx_aux - fx
    min_loc = np.argmin(fx_aux - fx)
    print("gap at tangent", gap[min_loc])
    print("majorizing ?", np.all(gap[:min_loc] > 0), np.all(gap[min_loc + 1 :]))

    ylim = [fx.min(), fx.max() * 3]

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, fx)
    ax.plot(x, fx_aux)
    ax.plot([x0], f(np.array([x0]), a, b, c, func=func), "x")
    ax.set_ylim(ylim)
    plt.show()
