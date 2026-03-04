"""
generate_presentation.py
Builds week08_presentation.ipynb — a 60-minute RISE/reveal.js slide deck
introducing Deep Learning and bridging into the MLP lab (Part 1).

Run:
    python generate_presentation.py

Present (needs RISE or nbconvert):
    jupyter nbconvert --to slides week08_presentation.ipynb --post serve
    # or install RISE and click "Enter/Exit RISE Slideshow" in Jupyter Lab/Notebook
"""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

FOLDER  = os.path.dirname(os.path.abspath(__file__))
OUTPUT  = os.path.join(FOLDER, "week08_presentation.ipynb")

# ---------------------------------------------------------------------------
# RISE / reveal.js notebook metadata
# ---------------------------------------------------------------------------
NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    },
    "rise": {
        "theme": "simple",
        "transition": "fade",
        "scroll": True,
        "enable_chalkboard": True,
        "slideNumber": True,
        "progress": True,
        "controls": True,
        "center": False,
        "width": 1400,
        "height": "95%",
        "minScale": 0.2,
        "maxScale": 2.0
    }
}

# ---------------------------------------------------------------------------
# Helper: create a slide cell (markdown or code) with RISE metadata
# ---------------------------------------------------------------------------

def slide(source, slide_type="slide"):
    """Markdown slide."""
    c = new_markdown_cell(source)
    c.metadata["slideshow"] = {"slide_type": slide_type}
    return c

def subslide(source):
    return slide(source, "subslide")

def fragment(source):
    return slide(source, "fragment")

def codeslide(source, slide_type="slide"):
    """Executable code cell on a slide."""
    c = new_code_cell(source)
    c.metadata["slideshow"] = {"slide_type": slide_type}
    return c

def codefrag(source):
    return codeslide(source, "fragment")

def notes(source):
    return slide(source, "notes")

# ---------------------------------------------------------------------------
# ── SECTION 0  Setup (hidden, slide_type=skip) ───────────────────────────
# ---------------------------------------------------------------------------

SETUP_CODE = """\
%%html
<style>
.reveal .cell { max-width: 100% !important; }
.reveal pre { width: 100% !important; max-width: 100% !important; }
.reveal .output_area pre { font-size: 0.9em; }
.reveal .jp-CodeCell .jp-Cell-inputWrapper { max-width: 100% !important; }
</style>

# Setup — run this cell first (hidden from slides)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 13,
})
"""

# ---------------------------------------------------------------------------
# ── SECTION 1  Introduction (slides 1-5, ~10 min) ────────────────────────
# ---------------------------------------------------------------------------

S1_TITLE = """\
<div style="text-align:center; padding-top:60px;">

# Deep Learning Fundamentals
## From Neurons to Networks

**MUSA 650 — Week 08**

*Introduction + Lab Part 1 Bridge*

---
⏱ 60 minutes &nbsp;|&nbsp; 3 parts &nbsp;|&nbsp; live code

</div>
"""

S1_AGENDA = """\
## Today's Agenda

| # | Topic | Time |
|---|---|---|
| 1 | What is Deep Learning? | ~10 min |
| 2 | Building Blocks of a Neural Network | ~15 min |
| 3 | How Networks Learn | ~15 min |
| 4 | MNIST & our MLP | ~10 min |
| 5 | Lab walkthrough | ~10 min |

> **Goal**: by the end you'll understand *every line of code* in Lab Part 1.
"""

S1_AI_ML_DL = """\
## AI ⊃ Machine Learning ⊃ Deep Learning

```
┌─────────────────────────────────────────────────┐
│  Artificial Intelligence                        │
│  ┌───────────────────────────────────────────┐  │
│  │  Machine Learning                         │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │  Deep Learning                      │  │  │
│  │  │  (representation learning)          │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

- **AI**: machines that perform tasks requiring human intelligence
- **ML**: systems that *learn from data* rather than being explicitly programmed
- **DL**: ML with many layers → learns *hierarchical representations* automatically
"""

S1_WHY_NOW = """\
## Why Deep Learning — and Why Now?

Three things converged around 2012:

| Factor | Explanation |
|---|---|
| 📦 **Data** | ImageNet: 14M labelled images; internet-scale text, satellite imagery |
| ⚡ **Compute** | GPUs parallelize matrix operations — 100× faster than CPUs |
| 🧠 **Algorithms** | Better initialisation, activation functions, regularisation |
| 🛠 **Frameworks** | TensorFlow, PyTorch, Keras — research → production in hours |

> AlexNet (2012) cut ImageNet error by almost **half** using a deep CNN trained on GPUs.
> That moment marked the beginning of the "deep learning era".
"""

S1_HISTORY = """\
## A Quick History

```
1943  McCulloch & Pitts  → first mathematical neuron model
1958  Rosenblatt         → Perceptron (single-layer learner)
1969  Minsky & Papert    → XOR problem — AI winter
1986  Rumelhart et al.   → Backpropagation rediscovered
1989  LeCun              → Convolutional Networks for digits
1990s                    → SVM dominates; DL in the background
2006  Hinton             → Deep Belief Networks, layer-wise pretraining
2012  Krizhevsky         → AlexNet wins ImageNet → DL revolution
2017  Attention          → Transformers (BERT, GPT, ...)
2020s                    → Foundation models, multimodal AI
```
"""

# ---------------------------------------------------------------------------
# ── SECTION 2  Building Blocks (~15 min) ─────────────────────────────────
# ---------------------------------------------------------------------------

S2_TITLE = """\
---
# Part 1 — Building Blocks of a Neural Network
"""

S2_BIO_NEURON = """\
## The Biological Inspiration

```
  Dendrites        Cell body          Axon
(receive inputs)  (integrates)    (fires output)

   x₁ ──┐
         │
   x₂ ──┤──► [ Σ + threshold ] ──► output signal
         │
   x₃ ──┘
```

Key ideas borrowed from biology:
- **Inputs** → weighted signals arrive from other neurons
- **Integration** → soma sums the inputs
- **Threshold** → fires (outputs 1) only if sum exceeds a threshold
- **Connections** → strength of connections (synapses) can change = **learning**
"""

S2_PERCEPTRON = """\
## The Artificial Neuron

$$\\hat{y} = f\\!\\left(\\sum_{i} w_i x_i + b\\right) = f(\\mathbf{w}^\\top \\mathbf{x} + b)$$

| Symbol | Name | Role |
|---|---|---|
| $x_i$ | **Input** | one feature value |
| $w_i$ | **Weight** | how much we trust that input |
| $b$ | **Bias** | shifts the threshold |
| $f(\\cdot)$ | **Activation** | adds non-linearity |
| $\\hat{y}$ | **Output** | the neuron's prediction |

> The **weights and bias** are the *learnable* parameters.
> Everything else is fixed architecture.
"""

S2_ACTIVATION_THEORY = """\
## Activation Functions — Why Non-Linearity?

Without an activation function every layer is just:
$$h = W_2 (W_1 x + b_1) + b_2 = W' x + b'$$

→ No matter how many layers, the whole network collapses to a **single linear transformation**.

Non-linear activations let us model complex, curved decision boundaries.

"""

S2_ACT_FUNC = """\
## Activation Functions

| Activation | Formula | Use |
|---|---|---|
| **ReLU** | max(0, z) | Default for hidden layers |
| **Sigmoid** | 1 / (1 + e⁻ᶻ) | Binary output, old-school hidden |
| **Softmax** | eᶻⁱ / Σⱼ eᶻʲ | Multi-class output (probabilities) |
| **Tanh** | (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ) | RNNs, centred around 0 |
"""

S2_ACTIVATION_CODE = """\
z = np.linspace(-4, 4, 200)

relu    = np.maximum(0, z)
sigmoid = 1 / (1 + np.exp(-z))
tanh    = np.tanh(z)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, y, name, color in zip(
        axes,
        [relu, sigmoid, tanh],
        ['ReLU  max(0, z)', 'Sigmoid  1/(1+e⁻ᶻ)', 'Tanh'],
        ['#e74c3c', '#3498db', '#2ecc71']):
    ax.plot(z, y, color=color, lw=2.5)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('z')
    ax.set_ylabel('f(z)')

plt.suptitle('Common Activation Functions', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""

S2_MLP_ARCH = """\
## From Neuron to Multi-Layer Perceptron (MLP)

```
Input layer     Hidden layer    Output layer
(784 pixels)    (512 neurons)   (10 classes)

  x₁ ──┐            h₁ ──┐          ŷ₀  (prob digit=0)
  x₂ ──┤──[W₁,b₁]──>h₂ ──┤──[W₂,b₂]──> ŷ₁  (prob digit=1)
  x₃ ──┤            h₃ ──┤          ...
  ... ──┤            ... ──┤          ŷ₉  (prob digit=9)
x₇₈₄ ──┘          h₅₁₂ ──┘
```

- Every neuron in layer $l$ connects to **every** neuron in layer $l+1$
  → called **Fully Connected** or **Dense** layer
- **Layer 1**: $H = \\text{ReLU}(W_1 X + b_1)$, shape $512$
- **Layer 2**: $\\hat{Y} = \\text{Softmax}(W_2 H + b_2)$, shape $10$
"""

S2_SOFTMAX = """\
## Softmax — Turning Scores into Probabilities

The output layer has **10 raw scores** (called *logits*), one per digit.

$$\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_{j=0}^{9} e^{z_j}}$$

Properties:
- All outputs are in $[0, 1]$
- They sum to exactly 1 → valid probability distribution
- The **largest logit gets the highest probability** (winner-takes-most)

```python
logits = [1.2, 0.3, 5.1, ...]   # raw model output
probs  = softmax(logits)         # [0.02, 0.01, 0.84, ...]
                                 # → predicts class 2 with 84% confidence
```
"""

# ---------------------------------------------------------------------------
# ── SECTION 3  Training (~15 min) ────────────────────────────────────────
# ---------------------------------------------------------------------------

S3_TITLE = """\
---
# Part 2 — How Networks Learn
"""

S3_LEARNING_OVERVIEW = """\
## The Learning Loop

```
          ┌─────────────────────────────────────────┐
          │                                         │
  Input ──►  Forward pass  ──►  Loss  ──►  Backward pass
  data       (prediction)        ↑           (update W, b)
                                 │                  │
                            How wrong?         Gradient
                                               descent
          └─────────────────────────────────────────┘
                    repeat for N epochs
```

Three questions to answer:
1. **How do we measure "wrong"?** → Loss function
2. **Which direction do we adjust the weights?** → Gradient (backprop)
3. **How big a step do we take?** → Learning rate / optimiser
"""

S3_LOSS = """\
## Loss Functions

### Regression → Mean Squared Error
$$\\mathcal{L}_{\\text{MSE}} = \\frac{1}{n}\\sum_i (y_i - \\hat{y}_i)^2$$

### Classification → Categorical Cross-Entropy  ← *we use this*
$$\\mathcal{L}_{\\text{CE}} = -\\sum_{k} y_k \\log(\\hat{y}_k)$$

- $y_k$: true label (one-hot: 1 for correct class, 0 for others)
- $\\hat{y}_k$: predicted probability for class $k$

> **Intuition**: heavily penalises *confident wrong* predictions.
> If $\\hat{y}_{\\text{correct}} = 0.01$, then $-\\log(0.01) = 4.6$ — a big loss!
> If $\\hat{y}_{\\text{correct}} = 0.99$, then $-\\log(0.99) ≈ 0.01$ — tiny loss.
"""

S3_LOSS_CODE = """\
eps = 1e-9
probs = np.linspace(eps, 1 - eps, 300)
loss  = -np.log(probs)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(probs, loss, color='#e74c3c', lw=2.5)
ax.fill_between(probs, loss, alpha=0.08, color='#e74c3c')
ax.set_xlabel('Predicted probability for the CORRECT class  p̂', fontsize=12)
ax.set_ylabel('Cross-Entropy Loss  −log(p̂)', fontsize=12)
ax.set_title('Cross-Entropy Loss', fontsize=14, fontweight='bold')
ax.axvline(0.5, color='gray', ls='--', lw=1, label='p̂ = 0.5  →  loss ≈ 0.69')
ax.axvline(0.9, color='#2ecc71', ls='--', lw=1, label='p̂ = 0.9  →  loss ≈ 0.11')
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
"""

S3_GD = """\
## Gradient Descent

Adjust each weight in the direction that **reduces the loss**:

$$w \\leftarrow w - \\eta \\cdot \\frac{\\partial \\mathcal{L}}{\\partial w}$$

- $\\eta$ (eta): **learning rate** — how big a step to take
- $\\partial \\mathcal{L}/\\partial w$: gradient — which direction is "uphill"

```
   Loss
    │    ●  ← starting point
    │      \\
    │       ●  ← after one step
    │        \\
    │         ●
    │          \\___●___●  ← converging to minimum
    └──────────────────── Weights
```

### Variants
| Name | Data per step | Notes |
|---|---|---|
| **Batch GD** | full dataset | stable, slow |
| **SGD** | 1 sample | noisy but fast |
| **Mini-batch GD** | `batch_size` samples | best of both ← *we use this* |
"""

S3_BACKPROP = """\
## Backpropagation

The **chain rule** of calculus lets us compute gradients for every weight,
starting from the loss and working backwards through the network.

```
  Forward:    x → h → ŷ → L
                             ↓
  Backward:   ∂L/∂W₁ ← ∂L/∂h · ∂h/∂W₁
```

Key facts:
- Backprop is just **repeated application of the chain rule**
- Frameworks (Keras/TF/PyTorch) do this **automatically** via *autograd*
- You almost never need to implement it by hand

> `model.fit(...)` calls forward pass + backprop for you at each batch.
"""

S3_OPTIMIZERS = """\
## Optimisers — Smarter than Plain Gradient Descent

| Optimiser | Key idea | When to use |
|---|---|---|
| **SGD** | Fixed learning rate | Simple baselines |
| **SGD + Momentum** | Accumulates velocity | Better convergence |
| **RMSprop** | Per-parameter adaptive LR | RNNs, our MLP lab ✓ |
| **Adam** | Momentum + RMSprop | Default choice for most tasks |
| **Adadelta** | Adaptive, no global LR | Our CNN lab ✓ |

**RMSprop** (we use in Part 1):
$$w \\leftarrow w - \\frac{\\eta}{\\sqrt{E[g^2] + \\epsilon}} \\cdot g$$

- Divides by the **running average of squared gradients**
- Prevents the learning rate from exploding or vanishing per parameter
"""

S3_OVERFITTING = """\
## Overfitting vs Underfitting

```
  Training accuracy  ████████████████████  98%
  Validation accuracy ████████████         72%   ← OVERFIT

  Training accuracy  ██████████            60%
  Validation accuracy █████████            58%   ← UNDERFIT (both bad)

  Training accuracy  ████████████████      92%
  Validation accuracy ████████████████     90%   ← GOOD
```

### Causes of overfitting
- Too many parameters relative to training data
- Too many epochs (model memorises noise)

### Solutions
- **More data** (or data augmentation)
- **Dropout** — randomly disable neurons during training
- **Early stopping** — halt when validation loss stops improving
- **Weight regularisation** (L1 / L2)
"""

S3_DROPOUT = """\
## Dropout — Our Main Regulariser

During each training step, randomly **zero out** a fraction $p$ of neurons:

```
Without Dropout:           With Dropout (p=0.5):
  h₁ ─────────────          h₁ ─── ✗  (dropped)
  h₂ ─────────────          h₂ ──────────────
  h₃ ─────────────          h₃ ─── ✗  (dropped)
  h₄ ─────────────          h₄ ──────────────
```

Why it works:
- Forces different neurons to learn **redundant representations**
- Equivalent to training an **ensemble** of $2^n$ sub-networks
- Dropout is **turned off at inference time** (weights scaled instead)

In the MLP lab we use `Dropout(0.2)` (optional, commented out).
In the CNN lab we use `Dropout(0.25)` and `Dropout(0.5)`.
"""

S3_EPOCHS_BATCHES = """\
## Epochs, Batches, and Iterations

| Term | Definition |
|---|---|
| **Sample** | One training example (one image) |
| **Batch** | A subset of samples processed together |
| **Iteration** | One forward + backward pass on one batch |
| **Epoch** | One full pass through the *entire* training set |

```
60 000 training images, batch_size = 20 000
→  3 iterations per epoch

epochs = 10  →  30 total iterations
```

**Larger batch**:  more stable gradient, uses more memory, fewer updates per epoch
**Smaller batch**: noisier gradient, less memory, more updates — often better generalisation
"""

# ---------------------------------------------------------------------------
# ── SECTION 4  MNIST & MLP (~10 min) ─────────────────────────────────────
# ---------------------------------------------------------------------------

S4_TITLE = """\
---
# Part 3 — MNIST and Our MLP
"""

S4_MNIST_INTRO = """\
## The MNIST Dataset

- **70 000** greyscale images of handwritten digits (0–9)
- **28 × 28** pixels → 784 input features per image
- **60 000** training / **10 000** test samples
- 10 balanced classes (~6 000 images per digit)

```
Training set:   ████████████████████████████████████████████████████████  60 000
Test set:       ██████████                                                 10 000
```

> MNIST is the **"Hello World" of deep learning**.
> If your model can't learn MNIST, something is fundamentally wrong.
> If it does learn MNIST, you're ready for harder datasets.
"""

S4_MNIST_VIZ_CODE = """\
from keras.datasets import mnist
(x_train_raw, y_train_raw), _ = mnist.load_data()

fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for digit in range(10):
    idx = np.where(y_train_raw == digit)[0][0]
    for row, ax in enumerate(axes[:, digit]):
        idx2 = np.where(y_train_raw == digit)[0][row]
        ax.imshow(x_train_raw[idx2], cmap='binary', vmin=0, vmax=255)
        ax.axis('off')
        if row == 0:
            ax.set_title(str(digit), fontsize=14, fontweight='bold')

plt.suptitle('Two examples of each MNIST digit', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
print(f"x_train shape: {x_train_raw.shape}  (samples, height, width)")
print(f"Pixel value range: {x_train_raw.min()} – {x_train_raw.max()}")
"""

S4_PREPROCESSING = """\
## Preprocessing — Three Steps

### 1. Flatten  (28×28 → 784)
```python
x_train = x_train.reshape(60000, 784)
x_test  = x_test.reshape(10000, 784)
```
Dense layers expect a **1-D feature vector**, not a 2-D image.

### 2. Normalise  ([0, 255] → [0, 1])
```python
x_train = x_train.astype('float32') / 255
x_test  = x_test.astype('float32')  / 255
```
Smaller numbers → stable gradients, faster convergence.

### 3. One-Hot Encode the Labels  (integer → vector)
```python
y_train = np_utils.to_categorical(y_train, 10)
# e.g.  5 → [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```
Matches the 10 softmax outputs; enables cross-entropy loss.
"""

S4_PREPROCESS_CODE = """\
import numpy as np
from tensorflow.python.keras.utils import np_utils

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

# --- Step 1: flatten ---
x_train = x_train_raw.reshape(60000, 784)
x_test  = x_test_raw.reshape(10000, 784)

# --- Step 2: normalise ---
x_train = x_train.astype('float32') / 255
x_test  = x_test.astype('float32')  / 255

# --- Step 3: one-hot encode ---
y_train = np_utils.to_categorical(y_train_raw, 10)
y_test  = np_utils.to_categorical(y_test_raw,  10)

# Show what changed
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].imshow(x_train_raw[0].reshape(28, 28), cmap='binary')
axes[0].set_title(f'Raw image  (label = {y_train_raw[0]})', fontsize=12)
axes[0].axis('off')
axes[1].bar(range(10), y_train[0], color=['#e74c3c' if i == y_train_raw[0]
            else '#bdc3c7' for i in range(10)])
axes[1].set_xticks(range(10))
axes[1].set_xlabel('Digit class')
axes[1].set_ylabel('Value')
axes[1].set_title(f'One-hot label for digit {y_train_raw[0]}', fontsize=12)
plt.tight_layout()
plt.show()
"""

S4_ARCHITECTURE = """\
## Our MLP Architecture

```
Input           Hidden          Output
(784)           (512, ReLU)     (10, Softmax)

 ●              ●               ●  P(digit=0)
 ●   W₁(784×512) ●   W₂(512×10) ●  P(digit=1)
 ●  ──────────► ●  ──────────► ●  ...
 .              .               ●  P(digit=9)
 ●              ●

784 inputs      512 neurons     10 outputs
```

**Parameter count**:
- W₁: 784 × 512 = 401 408
- b₁:         512
- W₂: 512 × 10  =   5 120
- b₂:          10
- **Total: 407 050** trainable parameters
"""

S4_KERAS_CODE = """\
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# --- Build ---
model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    # Dropout(0.2),          # ← try uncommenting during lab
    Dense(10,  activation='softmax'),
])

# --- Compile ---
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy']
)

model.summary()
"""

S4_TRAINING_CODE = """\
history = model.fit(
    x_train, y_train,
    batch_size=20000,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test)
)
"""

S4_INTERPRET_OUTPUT = """\
## Reading the Training Output

```
Epoch 8/10
3/3 ─── loss: 0.3335 ─ accuracy: 0.9053 ─ val_loss: 0.3042 ─ val_accuracy: 0.9176
```

| Field | Meaning | Good sign |
|---|---|---|
| `loss` | Cross-entropy on *training* set | Should decrease |
| `accuracy` | Fraction correct on *training* set | Should increase |
| `val_loss` | Cross-entropy on *test* set | Should decrease (close to `loss`) |
| `val_accuracy` | Fraction correct on *test* set | Should be close to `accuracy` |

**Red flags**:
- `accuracy` ≫ `val_accuracy` → overfitting
- Both stuck low → underfitting (more epochs, bigger model)
- `val_loss` increasing while `loss` decreases → overfitting
"""

# ---------------------------------------------------------------------------
# ── SECTION 5  Lab Bridge (~10 min) ──────────────────────────────────────
# ---------------------------------------------------------------------------

S5_TITLE = """\
---
# Part 4 — Lab Walkthrough
"""

S5_LAB_OVERVIEW = """\
## Lab Part 1 at a Glance

Open **`week08_lab_student.ipynb`** — 11 steps, end-to-end.

| Step | Task | Concept |
|---|---|---|
| 1 | Imports | `keras`, `numpy`, `matplotlib` |
| 2 | Load MNIST | `mnist.load_data()` |
| 3 | Visualise a sample | `plt.imshow` |
| 4 | Preprocess | reshape + normalise |
| 5 | One-hot encode | `np_utils.to_categorical` |
| 6 | Build model | `Sequential` + `Dense` |
| 7 | Visualise model | `plot_model` |
| 8 | Compile | `loss`, `optimizer`, `metrics` |
| 9 | Train | `model.fit` |
| 10 | Evaluate | `model.evaluate` |
| 11 | Reflect | Discussion questions |

> Expected test accuracy after 10 epochs: **~90%**
"""

S5_BLANKS = """\
## What the Blanks Look Like

**Step 4 (Preprocess)**:
```python
x_train = x_train.reshape(___, ___)   # ← what numbers go here?
x_train = x_train.astype(___)         # ← what dtype?
x_train /= ___                        # ← what do we divide by?
```

**Step 6 (Build model)**:
```python
model.add(Dense(___, activation=___, input_shape=___))
model.add(Dense(___, activation=___))
```

**Step 9 (Train)**:
```python
history = model.fit(x_train, y_train,
    batch_size=___,
    epochs=___,
    validation_data=___)
```

> Every blank has been introduced in this lecture — you have all the pieces!
"""

S5_EXPERIMENT = """\
## Experiments to Try in Step 11

After completing the basic lab, explore:

1. **Add a second hidden layer** (the commented `Dense(512)` block)
   → Does accuracy improve? Does training take longer?

2. **Enable Dropout(0.2)**
   → Does overfitting decrease?

3. **Change `batch_size`** to 128 vs 20 000
   → How many iterations per epoch? Which trains faster?

4. **Reduce `epochs`** to 3
   → Can the model still learn well?

5. **Change the optimiser** to `Adam()`
   → How does convergence speed change?
"""

S5_CONNECT_CNN = """\
## Bridge to Part 2 — What Changes for CNNs?

| | MLP (Part 1) | CNN (Part 2) |
|---|---|---|
| Input shape | `(784,)` — flattened | `(28, 28, 1)` — 2D kept |
| Main layers | `Dense` | `Conv2D`, `MaxPooling2D` |
| Spatial info | Lost (flatten) | Preserved (conv filters) |
| Parameters | ~407 K | ~1.2 M |
| Epochs needed | 10 | 12+ |
| Final accuracy | ~90% | ~99%+ (with enough epochs) |

> The **preprocessing difference** is the most important concept:
> CNNs don't flatten; they process 2D structure directly.
"""

# ---------------------------------------------------------------------------
# ── SECTION 6  Takeaways ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------

S6_TAKEAWAYS = """\
## Key Takeaways

1. **Neural networks** are stacks of parameterised transformations.
2. **Training** = minimising a loss function via gradient descent + backprop.
3. **ReLU** for hidden layers; **Softmax** for multi-class output.
4. **Preprocess** your data: flatten, normalise, one-hot encode.
5. **Keras Sequential API** makes building and training simple.
6. **Validation loss** tells you whether the model generalises.
7. **Dropout** is a cheap and effective regulariser.
8. CNNs beat MLPs on images — but the MLP is the essential foundation.
"""

S6_RESOURCES = """\
## Resources

**Interactive visualisations**
- [Neural Network Playground (TF)](https://playground.tensorflow.org) — draw your own dataset, see learning live
- [3Blue1Brown — Neural Networks](https://youtu.be/aircAruvnKk) — beautiful visual intro

**Reading**
- [Fully Connected vs CNN (Medium)](https://medium.com/swlh/fully-connected-vs-convolutional-neural-networks-813ca7bc6ee5)
- Keras docs: [Sequential model](https://keras.io/guides/sequential_model/)
- [CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)

**Practice**
- Complete **Lab Part 1** (MLP) → Lab Part 2 (CNN)
- Try the [Keras MNIST tutorial](https://keras.io/examples/vision/mnist_convnet/)
"""

S6_QA = """\
<div style="text-align:center; padding-top:80px;">

# Questions?

---

Open `week08_lab_student.ipynb` and start with **Step 1**.

*Solutions are in `week08_lab_solutions.ipynb`*
*— but try each step first!*

</div>
"""

# ---------------------------------------------------------------------------
# Assemble the notebook
# ---------------------------------------------------------------------------

def build_presentation():
    nb = new_notebook()
    nb.metadata.update(NOTEBOOK_METADATA)

    cells = [
        # Setup (hidden)
        codeslide(SETUP_CODE, slide_type="skip"),

        # Section 1: Introduction
        slide(S1_TITLE),
        slide(S1_AGENDA),
        slide(S1_AI_ML_DL),
        slide(S1_WHY_NOW),
        slide(S1_HISTORY),

        # Section 2: Building Blocks
        slide(S2_TITLE),
        slide(S2_BIO_NEURON),
        slide(S2_PERCEPTRON),
        slide(S2_ACTIVATION_THEORY),
        slide(S2_ACT_FUNC),
        codeslide(S2_ACTIVATION_CODE),
        slide(S2_MLP_ARCH),
        slide(S2_SOFTMAX),

        # Section 3: Training
        slide(S3_TITLE),
        slide(S3_LEARNING_OVERVIEW),
        slide(S3_LOSS),
        codeslide(S3_LOSS_CODE),
        slide(S3_GD),
        slide(S3_BACKPROP),
        slide(S3_OPTIMIZERS),
        slide(S3_EPOCHS_BATCHES),
        slide(S3_OVERFITTING),
        slide(S3_DROPOUT),

        # Section 4: MNIST & MLP
        slide(S4_TITLE),
        slide(S4_MNIST_INTRO),
        codeslide(S4_MNIST_VIZ_CODE),
        slide(S4_PREPROCESSING),
        codeslide(S4_PREPROCESS_CODE),
        slide(S4_ARCHITECTURE),
        codeslide(S4_KERAS_CODE),
        notes("Run this cell to show students the model summary live. "
              "Point out the parameter counts and how 784→512→10 maps to the diagram."),
        codeslide(S4_TRAINING_CODE, slide_type="subslide"),
        slide(S4_INTERPRET_OUTPUT),

        # Section 5: Lab
        slide(S5_TITLE),
        slide(S5_LAB_OVERVIEW),
        slide(S5_BLANKS),
        slide(S5_EXPERIMENT),
        slide(S5_CONNECT_CNN),

        # Section 6: Takeaways
        slide(S6_TAKEAWAYS),
        slide(S6_RESOURCES),
        slide(S6_QA),
    ]

    nb['cells'] = cells
    return nb


if __name__ == "__main__":
    pres = build_presentation()
    nbformat.write(pres, OUTPUT)
    print("Wrote:", OUTPUT)
    print()
    print("To view as slides:")
    print("  1. RISE (recommended): open in Jupyter Notebook/Lab -> click 'Enter/Exit RISE Slideshow'")
    print("  2. nbconvert: jupyter nbconvert --to slides week08_presentation.ipynb --post serve")
