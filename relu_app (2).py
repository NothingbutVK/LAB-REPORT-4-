"""ReLU activation visualiser + neural network demo (BSD3513 Lab Report - Neural Networks)."""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

NAME = "ReLU"
SKLEARN_ACTIVATION = "relu"  # activation name used by scikit-learn's MLPClassifier


def activation(x):
    """ReLU function."""
    x = np.asarray(x, dtype=float)
    return np.maximum(0, x)


def derivative(x):
    """Derivative of ReLU (supplies the gradient in backpropagation)."""
    x = np.asarray(x, dtype=float)
    return (x > 0).astype(float)


@st.cache_data(show_spinner=False)
def make_data(kind, n_samples, noise, seed=42):
    if kind == "Moons":
        return make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    if kind == "Circles":
        return make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    return make_blobs(n_samples=n_samples, centers=2, cluster_std=1 + noise * 3, random_state=seed)


def parse_layers(text):
    try:
        sizes = tuple(int(s) for s in text.split(",") if s.strip())
        if sizes and all(1 <= s <= 256 for s in sizes) and len(sizes) <= 5:
            return sizes
    except ValueError:
        pass
    return None


st.set_page_config(page_title=f"{NAME} Activation", page_icon="📈", layout="wide")
st.title(f"{NAME} Activation Function Explorer")
st.latex(r"\\text{ReLU}(x)=\\max(0,x)")

tab_curve, tab_nn = st.tabs(["📈 Activation curve", "🧠 Neural network demo"])

# ---------------------------------------------------------------- Tab 1
with tab_curve:
    left, right = st.columns([1, 2])
    with left:
        x_min, x_max = st.slider("x range", -20.0, 20.0, (-6.0, 6.0), step=0.5)
        x_in = st.number_input("Evaluate at x", value=1.0, step=0.5)
        show_deriv = st.checkbox("Show derivative", value=True)
        m1, m2 = st.columns(2)
        m1.metric("f(x)", f"{float(activation(x_in)):.4f}")
        m2.metric("f'(x)", f"{float(derivative(x_in)):.4f}")
        st.markdown("""- **Range:** [0, ∞)
- Cheap to compute; reduces vanishing gradients
- Weakness: 'dying ReLU' (zero gradient for x < 0)""")
    with right:
        if x_min >= x_max:
            st.error("Minimum x must be smaller than maximum x.")
        else:
            x = np.linspace(x_min, x_max, 500)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x, activation(x), label=f"{NAME}(x)", linewidth=2)
            if show_deriv:
                ax.plot(x, derivative(x), "--", label="derivative", linewidth=2)
            ax.scatter([x_in], [activation(x_in)], color="red", zorder=5, label=f"x = {x_in}")
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.axvline(0, color="grey", linewidth=0.5)
            ax.set_xlabel("x (weighted sum + bias)")
            ax.set_ylabel("output")
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)

# ---------------------------------------------------------------- Tab 2
with tab_nn:
    st.write(
        f"Train a small multi-layer perceptron whose hidden layers all use **{NAME}**, "
        "and watch how it separates two classes of points."
    )
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Dataset", ["Moons", "Circles", "Blobs"])
    n_samples = c1.slider("Samples", 100, 1000, 300, step=50)
    noise = c2.slider("Noise", 0.0, 0.5, 0.2, step=0.05)
    layers_txt = c2.text_input("Hidden layer sizes (comma-separated)", "16,16")
    max_iter = c3.slider("Max iterations", 100, 2000, 800, step=100)
    lr = c3.select_slider("Learning rate", options=[0.0005, 0.001, 0.005, 0.01, 0.05], value=0.01)

    layers = parse_layers(layers_txt)
    if layers is None:
        st.error("Enter 1-5 whole numbers between 1 and 256, e.g. 16,16")
        st.stop()

    X, y = make_data(kind, n_samples, noise)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    if st.button("Train network", type="primary"):
        with st.spinner("Training..."):
            model = MLPClassifier(
                hidden_layer_sizes=layers, activation=SKLEARN_ACTIVATION,
                learning_rate_init=lr, max_iter=max_iter, random_state=0,
            )
            model.fit(X_tr, y_tr)

        a1, a2, a3 = st.columns(3)
        a1.metric("Train accuracy", f"{model.score(X_tr, y_tr):.1%}")
        a2.metric("Test accuracy", f"{model.score(X_te, y_te):.1%}")
        a3.metric("Iterations used", model.n_iter_)

        gx, gy = np.meshgrid(
            np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
            np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200),
        )
        zz = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)

        p1, p2 = st.columns(2)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.contourf(gx, gy, zz, alpha=0.25, cmap="coolwarm")
        ax1.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="coolwarm", edgecolor="k", s=25, label="train")
        ax1.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", edgecolor="k", marker="s", s=35, label="test")
        ax1.set_title("Decision boundary")
        ax1.legend()
        p1.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(model.loss_curve_)
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("training loss")
        ax2.set_title("Loss curve")
        ax2.grid(alpha=0.3)
        p2.pyplot(fig2)

        n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
        st.caption(f"Architecture: 2 inputs -> {' -> '.join(map(str, layers))} -> 1 output | trainable parameters: {n_params}")
    else:
        st.info("Choose your settings, then press **Train network**.")
