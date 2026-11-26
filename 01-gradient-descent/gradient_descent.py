"""
Gradient Descent from scratch — Decoded ML, Article 01.

We fit a straight line  y = w * x + b  to noisy data using nothing but NumPy.
No scikit-learn, no PyTorch: just the "roll downhill" loop.
"""

import numpy as np


def make_data(n=50, true_w=3.0, true_b=2.0, noise=1.0, seed=42):
    """Create noisy points around the line y = true_w * x + true_b."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 5, n)
    y = true_w * x + true_b + rng.normal(0, noise, n)
    return x, y


def loss(w, b, x, y):
    """Mean Squared Error: how 'high up the hill' we are."""
    y_pred = w * x + b
    return np.mean((y_pred - y) ** 2)


def gradients(w, b, x, y):
    """Slope of the hill in the w-direction and the b-direction."""
    error = (w * x + b) - y
    dw = 2 * np.mean(error * x)
    db = 2 * np.mean(error)
    return dw, db


def gradient_descent(x, y, lr=0.01, steps=500, w=0.0, b=0.0, verbose=True):
    """
    The whole algorithm:
        1. Feel the slope       (compute gradients)
        2. Step downhill        (parameter = parameter - lr * gradient)
        3. Repeat
    Returns the final (w, b) and the history of every step.
    """
    history = []
    for step in range(steps):
        current_loss = loss(w, b, x, y)
        history.append((w, b, current_loss))

        dw, db = gradients(w, b, x, y)
        w = w - lr * dw
        b = b - lr * db

        if verbose and step % 50 == 0:
            print(f"step {step:3d} | loss {current_loss:8.4f} | w {w:6.3f} | b {b:6.3f}")

    history.append((w, b, loss(w, b, x, y)))
    return w, b, history


if __name__ == "__main__":
    x, y = make_data()
    w, b, history = gradient_descent(x, y)
    print(f"\nLearned line:  y = {w:.2f} * x + {b:.2f}")
    print("True line:     y = 3.00 * x + 2.00")
    print(f"Final loss:    {history[-1][2]:.4f}")
