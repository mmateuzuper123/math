import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Erdos Problem #1002 Analyzer", layout="wide")

st.title("Erdos Problem #1002 Computational Search Engine")
st.markdown(
    r"""
**Target Conjecture:** For $0 < \alpha < 1$, let 
$$f(\alpha, n) = \frac{1}{\log n} \sum_{1 \le k \le n} \left(\frac{1}{2} - \{\alpha k\}\right)$$
Does $f(\alpha, n)$ possess an asymptotic distribution function $g(c)$ as $n \to \infty$?
"""
)

# =========================
# CONTROLES DA BARRA LATERAL
# =========================
st.sidebar.header("Simulation Parameters")
num_alphas = st.sidebar.slider(
    "Sample Size of Alphas (N_alpha)", min_value=500, max_value=10000, value=2000, step=500
)
n_max = st.sidebar.slider(
    "Max Sequence Length (n)", min_value=1000, max_value=50000, value=10000, step=1000
)

# =========================
# MOTOR DE COMPUTACAO
# =========================
@st.cache_data
def compute_erdos_distribution(num_alphas, n_max):
    alphas = np.linspace(0.001, 0.999, num_alphas)
    k = np.arange(1, n_max + 1, dtype=np.float64)[:, None]
    ak = k * alphas[None, :]
    fractional_parts = ak - np.floor(ak)
    terms = 0.5 - fractional_parts
    cum_sums = np.cumsum(terms, axis=0)
    log_n = np.log(np.arange(1, n_max + 1, dtype=np.float64))[:, None]
    
    f_matrix = cum_sums / log_n
    final_values = f_matrix[-1, :]
    
    return alphas, final_values, f_matrix

if st.button("Run Empirical Distribution Search"):
    with st.spinner("Executing high-density matrix evaluations..."):
        alphas, final_vals, f_matrix = compute_erdos_distribution(num_alphas, n_max)
        
        st.success("Simulation complete.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min Value", f"{np.min(final_vals):.4f}")
        col2.metric("Max Value", f"{np.max(final_vals):.4f}")
        col3.metric("Mean", f"{np.mean(final_vals):.4f}")
        col4.metric("Standard Deviation", f"{np.std(final_vals):.4f}")
        
        st.subheader("Empirical Cumulative Distribution Function g_n(c)")
        
        sorted_vals = np.sort(final_vals)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sorted_vals, ecdf, label=f"n = {n_max}", color="blue", lw=2)
        ax.set_title("Candidate Asymptotic Distribution Function g(c)")
        ax.set_xlabel("c")
        ax.set_ylabel("g(c)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
        
        st.subheader("Trajectory of f(alpha, n) for Sample Alphas over n")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        log_x = np.log(np.arange(1, n_max + 1))
        
        for i in range(0, min(50, num_alphas), 5):
            ax2.plot(log_x, f_matrix[:, i], alpha=0.3, color="gray")
            
        ax2.set_title("Evolution of f(alpha, n) relative to log(n)")
        ax2.set_xlabel("log(n)")
        ax2.set_ylabel("f(alpha, n)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig2)
