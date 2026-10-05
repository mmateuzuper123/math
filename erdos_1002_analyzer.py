import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

st.set_page_config(page_title="Erdos Problem #1002 Active Search Engine", layout="wide")

st.title("Erdos Problem #1002 Active Computational Search Engine")
st.markdown(
    r"""
**Target Conjecture:** For $0 < \alpha < 1$, let 
$$f(\alpha, n) = \frac{1}{\log n} \sum_{1 \le k \le n} \left(\frac{1}{2} - \{\alpha k\}\right)$$
Este painel combina a distribuição empírica com **otimização global (SciPy)** para caçar ativamente os valores extremos de $\alpha$ que maximizam e minimizam $f(\alpha, n)$.
"""
)

# =========================
# CONTROLES DA BARRA LATERAL
# =========================
st.sidebar.header("Simulation Parameters")
num_alphas = st.sidebar.slider(
    "Sample Size of Alphas (N_alpha)", min_value=500, max_value=5000, value=1500, step=500
)
n_max = st.sidebar.slider(
    "Max Sequence Length (n)", min_value=1000, max_value=30000, value=5000, step=1000
)

# =========================
# FUNÇÕES DE CÁLCULO E BUSCA
# =========================
def evaluate_f(alpha, n):
    k = np.arange(1, n + 1, dtype=np.float64)
    ak = k * alpha
    fractional_parts = ak - np.floor(ak)
    terms = 0.5 - fractional_parts
    return np.sum(terms) / np.log(n)

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

if st.button("Run Empirical Search & Active Optimization"):
    with st.spinner("Executando varredura e algoritmos de otimização global..."):
        alphas, final_vals, f_matrix = compute_erdos_distribution(num_alphas, n_max)
        
        # -----------------
        # BUSCA ATIVA (SCIPI)
        # -----------------
        # Otimizador para encontrar o mínimo global de f(alpha, n_max)
        obj_func = lambda x: evaluate_f(x[0], n_max)
        
        res_min = differential_evolution(obj_func, bounds=[(0.001, 0.999)], seed=42, maxiter=20, popsize=15)
        # Para achar o máximo, minimizamos o negativo da função
        obj_func_neg = lambda x: -evaluate_f(x[0], n_max)
        res_max = differential_evolution(obj_func_neg, bounds=[(0.001, 0.999)], seed=42, maxiter=20, popsize=15)
        
        opt_min_alpha = res_min.x[0]
        opt_min_val = evaluate_f(opt_min_alpha, n_max)
        
        opt_max_alpha = res_max.x[0]
        opt_max_val = evaluate_f(opt_max_alpha, n_max)
        
        st.success("Simulação e Otimização concluídas.")
        
        # Métricas gerais e de otimização
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mínimo Encontrado (Grid)", f"{np.min(final_vals):.4f}")
        col2.metric("Máximo Encontrado (Grid)", f"{np.max(final_vals):.4f}")
        col3.metric("Mínimo Otimizado (SciPy)", f"{opt_min_val:.4f} (α={opt_min_alpha:.4f})")
        col4.metric("Máximo Otimizado (SciPy)", f"{opt_max_val:.4f} (α={opt_max_alpha:.4f})")
        
        # ECDF Plot
        st.subheader("Empirical Cumulative Distribution Function g_n(c)")
        sorted_vals = np.sort(final_vals)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sorted_vals, ecdf, label=f"n = {n_max}", color="blue", lw=2)
        ax.axvline(opt_min_val, color="red", linestyle=":", label=f"Global Min (α={opt_min_alpha:.3f})")
        ax.axvline(opt_max_val, color="green", linestyle=":", label=f"Global Max (α={opt_max_alpha:.3f})")
        ax.set_title("Candidate Asymptotic Distribution Function g(c) with Optimized Extremes")
        ax.set_xlabel("c")
        ax.set_ylabel("g(c)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
        
        # Trajectory Evolution Plot
        st.subheader("Trajectory Evolution for Sample Alphas & Optimized Extremes")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        log_x = np.log(np.arange(1, n_max + 1))
        
        for i in range(0, min(50, num_alphas), 5):
            ax2.plot(log_x, f_matrix[:, i], alpha=0.2, color="gray")
            
        # Trajetória dos extremos otimizados
        opt_min_traj = np.array([evaluate_f(opt_min_alpha, step) for step in range(10, n_max+1, max(1, n_max//100))])
        opt_max_traj = np.array([evaluate_f(opt_max_alpha, step) for step in range(10, n_max+1, max(1, n_max//100))])
        steps_sampled = np.log(np.arange(10, n_max+1, max(1, n_max//100)))
        
        ax2.plot(steps_sampled, opt_min_traj, color="red", lw=2, label="Optimized Min Trajectory")
        ax2.plot(steps_sampled, opt_max_traj, color="green", lw=2, label="Optimized Max Trajectory")
        
        ax2.set_title("Evolution of f(alpha, n) including Extreme Search Paths")
        ax2.set_xlabel("log(n)")
        ax2.set_ylabel("f(alpha, n)")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig2)
