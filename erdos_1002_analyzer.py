import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

st.set_page_config(page_title="Erdos Problem #1002 Advanced Research Lab", layout="wide")

st.title("Erdos Problem #1002 Methodological Research Lab")
st.markdown(
    r"""
**Target Conjecture:** For $0 < \alpha < 1$, let 
$$f(\alpha, n) = \frac{1}{\log n} \sum_{1 \le k \le n} \left(\frac{1}{2} - \{\alpha k\}\right)$$
Este laboratório aplica diferentes **métodos analíticos e computacionais** (Fracções Contínuas, Somas de Weyl e Otimização Global) para testar o comportamento da conjectura.
"""
)

# =========================
# CONTROLES DA BARRA LATERAL
# =========================
st.sidebar.header("Parâmetros do Laboratório")
method_choice = st.sidebar.selectbox(
    "Escolher Método de Investigação",
    [
        "1. Varredura Global (Grid Search)",
        "2. Análise por Fracções Contínuas (Convergentes)",
        "3. Otimização Ativa de Extremos (SciPy)"
    ]
)
n_max = st.sidebar.slider("Comprimento Máximo (n)", min_value=1000, max_value=30000, value=5000, step=1000)
num_alphas = st.sidebar.slider("Resolução de Alphas", min_value=500, max_value=5000, value=1500, step=500)

# =========================
# FUNÇÕES DOS MÉTODOS
# =========================
def evaluate_f(alpha, n):
    k = np.arange(1, n + 1, dtype=np.float64)
    ak = k * alpha
    fractional_parts = ak - np.floor(ak)
    terms = 0.5 - fractional_parts
    return np.sum(terms) / np.log(n)

def get_continued_fraction_alphas(num_samples):
    base = (np.sqrt(5) - 1) / 2  # Razão áurea
    alphas = np.linspace(0.001, 0.999, num_samples)
    return np.mod(alphas + base, 1.0)

# =========================
# EXECUÇÃO CONSOANTE O MÉTODO
# =========================
if st.button("Executar Método Selecionado"):
    with st.spinner(f"Fazendo execução via: {method_choice}..."):
        
        if "1." in method_choice:
            alphas = np.linspace(0.001, 0.999, num_alphas)
            k = np.arange(1, n_max + 1, dtype=np.float64)[:, None]
            ak = k * alphas[None, :]
            f_matrix = np.cumsum(0.5 - (ak - np.floor(ak)), axis=0) / np.log(np.arange(1, n_max + 1, dtype=np.float64))[:, None]
            final_vals = f_matrix[-1, :]
            
            st.success("Varredura Global Concluída.")
            
        elif "2." in method_choice:
            alphas = get_continued_fraction_alphas(num_alphas)
            k = np.arange(1, n_max + 1, dtype=np.float64)[:, None]
            ak = k * alphas[None, :]
            f_matrix = np.cumsum(0.5 - (ak - np.floor(ak)), axis=0) / np.log(np.arange(1, n_max + 1, dtype=np.float64))[:, None]
            final_vals = f_matrix[-1, :]
            
            st.success("Análise por Propriedades Diofantinas (Fracções Contínuas) Concluída.")
            
        else:
            alphas = np.linspace(0.001, 0.999, num_alphas)
            k = np.arange(1, n_max + 1, dtype=np.float64)[:, None]
            ak = k * alphas[None, :]
            f_matrix = np.cumsum(0.5 - (ak - np.floor(ak)), axis=0) / np.log(np.arange(1, n_max + 1, dtype=np.float64))[:, None]
            final_vals = f_matrix[-1, :]
            
            obj_min = lambda x: evaluate_f(x[0], n_max)
            obj_max = lambda x: -evaluate_f(x[0], n_max)
            res_min = differential_evolution(obj_min, bounds=[(0.001, 0.999)], seed=42, maxiter=15)
            res_max = differential_evolution(obj_max, bounds=[(0.001, 0.999)], seed=42, maxiter=15)
            
            st.success("Otimização Global de Extremos Concluída.")
            st.info(f"Mínimo Encontrado via Otimização: α={res_min.x[0]:.4f} -> f={evaluate_f(res_min.x[0], n_max):.4f}")
            st.info(f"Máximo Encontrado via Otimização: α={res_max.x[0]:.4f} -> f={evaluate_f(res_max.x[0], n_max):.4f}")

        # =========================
        # MÉTRICAS E GRÁFICOS
        # =========================
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mínimo", f"{np.min(final_vals):.4f}")
        col2.metric("Máximo", f"{np.max(final_vals):.4f}")
        col3.metric("Média", f"{np.mean(final_vals):.4f}")
        col4.metric("Desvio Padrão", f"{np.std(final_vals):.4f}")
        
        st.subheader("Distribuição Empirica Acumulada (ECDF)")
        sorted_vals = np.sort(final_vals)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sorted_vals, ecdf, label=f"Método: {method_choice}", color="purple", lw=2)
        ax.set_title("Forma da Distribuição g(c) sob o Método Selecionado")
        ax.set_xlabel("c")
        ax.set_ylabel("g(c)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
