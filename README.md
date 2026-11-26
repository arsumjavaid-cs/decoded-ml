# Decoded ML 🧩

**Hard machine learning concepts, decoded from scratch, one at a time.**

I'm a mechanical engineer turned AI/ML engineer. When I started learning machine learning, a lot of it looked like hieroglyphics: dense formulas, unfamiliar words, and tutorials that skipped the "why".

So I started *decoding* each concept the way an engineer would: find an analogy I already understand, strip the math down to what's actually needed, then **build it from scratch** until it clicks.

This repo holds the code behind my Medium series **"Decoded: ML the Way an Engineer Learns It."** Every article has a folder with:

- a from-scratch implementation (NumPy first, frameworks later)
- a Jupyter notebook you can run top to bottom
- the script that generates every diagram in the article

---

## 📚 The Series

| # | Concept | Article | Code | Status |
|---|---------|---------|------|--------|
| 01 | Gradient Descent | *Every AI You've Ever Used Learned by Rolling Downhill* | [01-gradient-descent](./01-gradient-descent) | 📝 Published |
| 02 | Overfitting & Bias–Variance | Coming soon | [02-overfitting](./02-overfitting) | 🔜 |
| 03 | Evaluation Metrics | Coming soon | [03-evaluation-metrics](./03-evaluation-metrics) | 🔜 |
| 04 | Backpropagation | Coming soon | [04-backpropagation](./04-backpropagation) | 🔜 |
| 05 | Convolutional Neural Networks | Coming soon | [05-convolutional-neural-networks](./05-convolutional-neural-networks) | 🔜 |
| 06 | Embeddings | Coming soon | [06-embeddings](./06-embeddings) | 🔜 |
| 07 | Attention & Transformers | Coming soon | [07-attention-and-transformers](./07-attention-and-transformers) | 🔜 |
| 08 | Retrieval-Augmented Generation (RAG) | Coming soon | [08-retrieval-augmented-generation](./08-retrieval-augmented-generation) | 🔜 |
| 09 | AI Agents | Coming soon | [09-ai-agents](./09-ai-agents) | 🔜 |
| 10 | ML in Production & Data Drift | Coming soon | [10-ml-in-production](./10-ml-in-production) | 🔜 |

> Replace "Coming soon" with the Medium link as each article goes live.

---

## 🚀 Run It Yourself

```bash
git clone https://github.com/arsumjavaid-cs/decoded-ml.git
cd decoded-ml

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run any article's code, e.g. Article 01
cd 01-gradient-descent
python gradient_descent.py       # trains the model, prints progress
python make_figures.py           # regenerates every diagram into images/
jupyter notebook gradient_descent.ipynb
```

---

## 🗂️ Repo Structure

```
decoded-ml/
├── README.md                  ← you are here
├── requirements.txt
├── LICENSE
├── assets/                    ← series banner, shared images
├── 01-gradient-descent/
│   ├── README.md              ← summary + link to the article
│   ├── gradient_descent.py    ← from-scratch implementation
│   ├── gradient_descent.ipynb ← step-by-step notebook
│   ├── make_figures.py        ← generates the article's diagrams
│   └── images/                ← output diagrams (PNG/GIF)
├── 02-overfitting/
│   └── ...same layout
└── ...
```

---

## 🧠 How Every Article Is Built

1. **The wall:** where I got stuck
2. **The decode:** one analogy that made it click
3. **The math, gently:** only what's needed, every symbol explained
4. **Build it:** from scratch, in this repo
5. **In the real world:** where it shows up in industry
6. **The takeaway:** one line to remember

---

## 👋 About Me

**Arsum Javaid**: Mechanical engineer (B.Tech + M.Tech) → AI/ML engineer · PG Diploma in AI/ML & Agentic AI Engineering, IIT Gandhinagar · FIDE National Instructor (chess)

- Medium: *add your link*
- LinkedIn: *add your link*
- GitHub: [@arsumjavaid-cs](https://github.com/arsumjavaid-cs)

If this helped you understand something, a ⭐ on the repo means a lot.
