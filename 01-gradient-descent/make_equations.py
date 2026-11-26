"""Renders the article's formulas as PNG images (Medium has no LaTeX support).
Run:  python make_equations.py   ->  images/equations/
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "cm"

OUT = Path(__file__).parent / "images" / "equations"
OUT.mkdir(parents=True, exist_ok=True)

EQUATIONS = {
    "eq1_update_rule_general": r"$\theta \;=\; \theta \;-\; \alpha\, \nabla J(\theta)$",
    "eq2_line_model": r"$\hat{y} \;=\; w \cdot x \;+\; b$",
    "eq3_mse_loss": r"$L(w, b) \;=\; \dfrac{1}{n} \sum_{i=1}^{n} \left( w x_i + b - y_i \right)^2$",
    "eq4_gradients": r"$\dfrac{\partial L}{\partial w} = \dfrac{2}{n} \sum_{i=1}^{n} (w x_i + b - y_i)\, x_i"
                     r"\qquad \dfrac{\partial L}{\partial b} = \dfrac{2}{n} \sum_{i=1}^{n} (w x_i + b - y_i)$",
    "eq5_update_rule_w_b": r"$w \leftarrow w - \alpha \dfrac{\partial L}{\partial w}"
                           r"\qquad b \leftarrow b - \alpha \dfrac{\partial L}{\partial b}$",
}

for name, tex in EQUATIONS.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=26, color="#1f2937")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.25,
                facecolor="white")
    plt.close(fig)
print(f"Saved {len(EQUATIONS)} equations to {OUT.resolve()}")
