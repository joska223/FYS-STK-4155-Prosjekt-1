import numpy as np
import matplotlib.pyplot as plt

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

# ------------------------------------------------------------
# Your data function
# ------------------------------------------------------------

def exercise_data(n=100, degree=2, sigma=0.5, seed=2026):
    """Standardised polynomial design matrix (no intercept column)
    and centred targets."""

    rng = np.random.default_rng(seed)

    x = np.sort(rng.uniform(-1, 1, n))
    y = runge(x) + rng.normal(0, sigma, n)

    X = np.column_stack([x**k for k in range(1, degree + 1)])

    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    return X_norm, y - y.mean()


# ------------------------------------------------------------
# Make shuffled minibatches
# ------------------------------------------------------------

def make_batches(n, batch_size, rng):
    """Shuffle the data and split it into minibatches."""

    indices = rng.permutation(n)

    return [
        indices[i:i + batch_size]
        for i in range(0, n, batch_size)
    ]


# ------------------------------------------------------------
# Analytical OLS solution
# ------------------------------------------------------------

def ols_closed_form(X, y):
    """Exact OLS solution."""

    return np.linalg.lstsq(X, y, rcond=None)[0]


# ------------------------------------------------------------
# SGD with different optimizers
# ------------------------------------------------------------

def sgd_ols(
    X,
    y,
    method="gd",
    batch_size=5,
    gamma=0.001,
    n_epochs=1000,
    beta=0.9,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    seed=2026
):
    """
    Minibatch stochastic gradient descent for OLS.

    method:
        "gd"        - plain gradient descent
        "momentum"  - momentum
        "adagrad"   - AdaGrad
        "rmsprop"   - RMSProp
        "adam"      - Adam

    Returns:
        theta
        history
    """

    n, p = X.shape

    rng = np.random.default_rng(seed)

    # Initial parameters
    theta = np.zeros(p)

    # Optimizer states
    momentum = np.zeros(p)
    first_moment = np.zeros(p)
    second_moment = np.zeros(p)

    # Adam update counter
    t = 0

    # Store the full-data loss once per epoch
    history = np.empty(n_epochs)

    for epoch in range(n_epochs):

        batches = make_batches(n, batch_size, rng)

        for indices in batches:

            Xb = X[indices]
            yb = y[indices]

            # ------------------------------------------------
            # Gradient of MSE:
            #
            # L = 1/m sum (X theta - y)^2
            #
            # grad L = 2/m X^T(X theta - y)
            # ------------------------------------------------

            residual = Xb @ theta - yb

            gradient = (
                2 / len(indices)
                * Xb.T @ residual
            )

            # ------------------------------------------------
            # Plain gradient descent
            # ------------------------------------------------

            if method == "gd":

                theta -= gamma * gradient

            # ------------------------------------------------
            # Momentum
            # ------------------------------------------------

            elif method == "momentum":

                momentum = (
                    beta * momentum
                    + (1 - beta) * gradient
                )

                theta -= gamma * momentum

            # ------------------------------------------------
            # AdaGrad
            # ------------------------------------------------

            elif method == "adagrad":

                second_moment += gradient**2

                theta -= (
                    gamma
                    * gradient
                    / (np.sqrt(second_moment) + eps)
                )

            # ------------------------------------------------
            # RMSProp
            # ------------------------------------------------

            elif method == "rmsprop":

                second_moment = (
                    beta * second_moment
                    + (1 - beta) * gradient**2
                )

                theta -= (
                    gamma
                    * gradient
                    / (np.sqrt(second_moment) + eps)
                )

            # ------------------------------------------------
            # Adam
            # ------------------------------------------------

            elif method == "adam":

                t += 1

                first_moment = (
                    beta1 * first_moment
                    + (1 - beta1) * gradient
                )

                second_moment = (
                    beta2 * second_moment
                    + (1 - beta2) * gradient**2
                )

                # Bias correction
                first_corrected = (
                    first_moment
                    / (1 - beta1**t)
                )

                second_corrected = (
                    second_moment
                    / (1 - beta2**t)
                )

                theta -= (
                    gamma
                    * first_corrected
                    / (np.sqrt(second_corrected) + eps)
                )

            else:
                raise ValueError(
                    "method must be one of: "
                    "'gd', 'momentum', 'adagrad', "
                    "'rmsprop', 'adam'"
                )

        # Full-data loss after this epoch
        history[epoch] = np.mean(
            (X @ theta - y)**2
        )

    return theta, history


# ------------------------------------------------------------
# Generate data
# ------------------------------------------------------------

X, y = exercise_data(
    n=100,
    degree=5,
    sigma=0.5,
    seed=2026
)


# ------------------------------------------------------------
# Exact analytical solution
# ------------------------------------------------------------

theta_exact = ols_closed_form(X, y)

loss_exact = np.mean(
    (X @ theta_exact - y)**2
)

print("Exact OLS theta:")
print(theta_exact)

print("\nExact OLS loss:")
print(loss_exact)


# ------------------------------------------------------------
# Test all five methods
# ------------------------------------------------------------

settings = {
    "gd": {
        "gamma": 0.0005
    },

    "momentum": {
        "gamma": 0.0005,
        "beta": 0.9
    },

    "adagrad": {
        "gamma": 0.01
    },

    "rmsprop": {
        "gamma": 0.0001,
        "beta": 0.999
    },

    "adam": {
        "gamma": 0.0003,
        "beta1": 0.9,
        "beta2": 0.999
    }
}


results = {}


for method, params in settings.items():

    theta, history = sgd_ols(
        X,
        y,
        method=method,
        batch_size=5,
        n_epochs=5000,
        seed=2026,
        **params
    )

    parameter_error = np.linalg.norm(
        theta - theta_exact
    )

    loss_error = history[-1] - loss_exact

    results[method] = {
        "theta": theta,
        "history": history,
        "parameter_error": parameter_error,
        "loss_error": loss_error
    }

    print("\n", method)
    print("theta =", theta)
    print("parameter error =", parameter_error)
    print("loss =", history[-1])
    print("loss error =", loss_error)


# ------------------------------------------------------------
# Plot convergence
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

for method in settings:

    plt.plot(
        results[method]["history"],
        label=method
    )

plt.axhline(
    loss_exact,
    linestyle="--",
    label="exact OLS"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()