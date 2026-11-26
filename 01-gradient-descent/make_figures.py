"""
Generates every diagram used in Article 01. Run:  python make_figures.py
Output goes to ./images/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from gradient_descent import gradient_descent, make_data

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

INK = "#1f2937"
ACCENT = "#e4572e"
GOOD = "#2a9d8f"
SOFT = "#9ca3af"
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def valley(t):
    """A simple 1-D 'hill': loss = (t - 3)^2 + 1."""
    return (t - 3) ** 2 + 1


def valley_grad(t):
    return 2 * (t - 3)


def descend_1d(start, lr, steps):
    pts = [start]
    for _ in range(steps):
        pts.append(pts[-1] - lr * valley_grad(pts[-1]))
    return np.array(pts)


# ---------------------------------------------------------------- Figure 1
def fig_ball_rolling():
    t = np.linspace(-1, 7, 300)
    pts = descend_1d(start=-0.5, lr=0.15, steps=10)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    ax.plot(t, valley(t), color=INK, lw=2.5)
    ax.plot(pts, valley(pts), "o", color=ACCENT, ms=9, zorder=3)
    for a, b in zip(pts[:-1], pts[1:]):
        ax.annotate("", xy=(b, valley(b)), xytext=(a, valley(a)),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    ax.annotate("Start: a random guess\n(high error)", xy=(pts[0], valley(pts[0])),
                xytext=(0.6, 16.5), fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="-", color=SOFT))
    ax.annotate("Bottom of the valley\n= lowest error", xy=(3, 1), xytext=(5.0, 2.2),
                fontsize=11, color=GOOD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GOOD))
    ax.text(1.0, 9.0, "big steps where\nthe slope is steep", fontsize=10, color=SOFT)
    ax.annotate("tiny steps\nnear the bottom", xy=(2.75, 1.4), xytext=(2.3, 4.2), fontsize=10, color=SOFT,
                arrowprops=dict(arrowstyle="-", color=SOFT))
    ax.set_xlabel("Model parameter (e.g. the slope of our line)")
    ax.set_ylabel("Error (loss)")
    ax.set_title("Gradient descent = a ball rolling downhill, one step at a time",
                 color=INK, fontweight="bold", loc="left")
    ax.set_ylim(0, 20)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_ball_rolling_downhill.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig_learning_rates():
    t = np.linspace(-1, 7, 300)
    cases = [
        ("Too small (lr = 0.02)", 0.02, 12, SOFT, "Crawls. Still far from\nthe bottom after 12 steps."),
        ("Just right (lr = 0.3)", 0.3, 12, GOOD, "Reaches the bottom\nquickly and settles."),
        ("Too big (lr = 1.05)", 1.05, 6, ACCENT, "Overshoots, bounces\nwider and wider. Diverges."),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150, sharey=True)
    for ax, (title, lr, steps, color, note) in zip(axes, cases):
        pts = descend_1d(start=0.0, lr=lr, steps=steps)
        ax.plot(t, valley(t), color=INK, lw=2)
        ax.plot(pts, valley(pts), "-o", color=color, ms=6, lw=1.4)
        ax.text(0.5, 1.16, title, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=13, color=color, fontweight="bold")
        ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color=INK)
        ax.set_xlim(-1, 7)
        ax.set_ylim(0, 20)
        ax.set_xlabel("Parameter")
    axes[0].set_ylabel("Error (loss)")
    fig.suptitle("The learning rate decides how big each step is",
                 fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_learning_rate_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3 + GIF
def fig_line_fitting():
    x, y = make_data()
    _, _, history = gradient_descent(x, y, lr=0.01, steps=500, verbose=False)
    xs = np.linspace(0, 5, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    ax1.scatter(x, y, color=INK, s=18, alpha=0.7, label="Data")
    for step, alpha in [(0, 0.3), (2, 0.45), (8, 0.65), (500, 1.0)]:
        w, b, _ = history[step]
        color = GOOD if step == 500 else ACCENT
        ax1.plot(xs, w * xs + b, color=color, alpha=alpha, lw=2,
                 label=f"step {step}: y = {w:.2f}x + {b:.2f}")
    ax1.set_title("The line improves with every step", fontweight="bold", loc="left")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(fontsize=9, frameon=False)

    losses = [h[2] for h in history]
    ax2.plot(losses, color=ACCENT, lw=2.2)
    ax2.set_yscale("log")
    ax2.set_title("The error falls quickly, then flattens", fontweight="bold", loc="left")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss (log scale)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_line_fitting_and_loss.png")
    plt.close(fig)

    # Animated GIF (great as the Medium cover / LinkedIn media)
    frames = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 80, 110, 150, 200, 280, 380, 500, 500, 500]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=110)
    ax.scatter(x, y, color=INK, s=18, alpha=0.7)
    (line,) = ax.plot([], [], color=ACCENT, lw=3)
    label = ax.text(0.03, 0.93, "", transform=ax.transAxes, fontsize=12, color=INK,
                    fontweight="bold")
    ax.set_xlim(0, 5)
    ax.set_ylim(min(y.min(), 0) - 1, y.max() + 2)
    ax.set_title("Gradient descent fitting a line, step by step", loc="left", fontweight="bold")

    def update(i):
        step = frames[i]
        w, b, l = history[step]
        line.set_data(xs, w * xs + b)
        line.set_color(GOOD if step == 500 else ACCENT)
        label.set_text(f"step {step:3d}   loss = {l:6.2f}")
        return line, label

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    anim.save(OUT / "fig4_line_fitting_animation.gif", writer=PillowWriter(fps=4))
    plt.close(fig)


if __name__ == "__main__":
    fig_ball_rolling()
    fig_learning_rates()
    fig_line_fitting()
    print(f"Saved figures to {OUT.resolve()}")
