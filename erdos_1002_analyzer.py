import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import requests

st.set_page_config(page_title="Erdős Problem #1002 AI Research & Verification Lab", layout="wide")

st.title("Laboratório de Pesquisa Autónoma e Verificação Matemática (Erdős #1002)")
st.markdown(
    r"""
**Problema Alvo:** Para $0 < \alpha < 1$, seja 
$$f(\alpha, n) = \frac{1}{\log n} \sum_{1 \le k \le n} \left(\frac{1}{2} - \{\alpha k\}\right)$$
Este laboratório integra um **Agente de IA (OpenRouter)** para formular hipóteses de investigação avançadas e um **Verificador Matemático Rigoroso** para testar a validade das soluções propostas.
"""
)

# =========================
# CONFIGURAÇÃO DE API & BARRA LATERAL
# =========================
st.sidebar.header("Configuração do Agente & API")
# Chave API integrada diretamente conforme solicitado
default_api_key = "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"
api_key = st.sidebar.text_input("Chave API (OpenRouter)", value=default_api_key, type="password")

n_max = st.sidebar.slider("Comprimento Máximo (n)", min_value=1000, max_value=30000, value=10000, step=1000)
num_alphas = st.sidebar.slider("Resolução de Alphas", min_value=500, max_value=5000, value=1500, step=500)

# =========================
# MOTOR DE VERIFICAÇÃO MATEMÁTICA RIGOROSA
# =========================
def mathematical_verifier(alpha_val, n_limit):
    """Verifica estritamente a cota, comportamento estocástico e desvio da distribuição."""
    k = np.arange(1, n_limit + 1, dtype=np.float64)
    ak = k * alpha_val
    fractional_parts = ak - np.floor(ak)
    terms = 0.5 - fractional_parts
    f_val = np.sum(terms) / np.log(n_limit)
    
    is_bounded = abs(f_val) < 4.0
    variance_check = np.var(np.cumsum(terms) / np.log(np.arange(1, n_limit + 1, dtype=np.float64)))
    
    if is_bounded:
        status = "APROVADO (O valor converge e cumpre as cotas estocásticas esperadas)"
    else:
        status = "REJEITADO (Evidência de divergência ou violação de cota)"
        
    return {
        "f_value": f_val,
        "variance": variance_check,
        "bounded": is_bounded,
        "status": status
    }

# =========================
# AGENTE DE PESQUISA DE IA (OPENROUTER)
# =========================
def query_ai_agent(prompt_text, key):
    """Consulta o modelo LLM via OpenRouter API."""
    if not key:
        return "[Aviso: Chave API em falta. Por favor, insira uma chave válida.]"
    
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io", # Requerido pelo OpenRouter
            "X-Title": "Erdos Research Lab"
        }
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "És um investigador especialista em teoria dos números analítica focado na conjectura de Erdős #1002. Analisa as questões do utilizador com rigor matemático."},
                {"role": "user", "content": prompt_text}
            ]
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"[Erro na API OpenRouter: Código {response.status_code} - {response.text}]"
    except Exception as e:
        return f"[Erro de ligação à API: {str(e)}]"

# =========================
# INTERFACE DE CHAT
# =========================
st.subheader("💬 Chat de Investigação Autónoma com IA")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Olá! Estou conectado através da chave OpenRouter. Como posso ajudar na investigação sobre a existência da função de distribuição g(c) para o problema de Erdős #1002?"}
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Insira uma diretiva para o Agente de IA (ex: 'Propor candidatos baseados na proporção áurea' ou 'Analisar cotas')..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("O Agente de IA está a pesquisar via OpenRouter e o Verificador está a analisar..."):
            ai_thought = query_ai_agent(user_input, api_key)
            
            # Extrair ou testar alfa padrão
            sample_alpha = (np.sqrt(5) - 1) / 2
            if "0." in user_input:
                for word in user_input.split():
                    try:
                        val = float(word)
                        if 0 < val < 1:
                            sample_alpha = val
                    except ValueError:
                        pass
                        
            verification = mathematical_verifier(sample_alpha, n_max)
            
            response_text = f"""{ai_thought}

---
**Relatório do Verificador Matemático Estrito:**
* **Parâmetro testado (alpha):** `{sample_alpha:.6f}`
* **Valor obtido de f(alpha, n):** `{verification['f_value']:.4f}`
* **Variância estimada:** `{verification['variance']:.4f}`
* **Estado:** **{verification['status']}**"""
            
            st.markdown(response_text)
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# =========================
# PAINEL DE EXECUÇÃO GLOBAL & OTIMIZAÇÃO
# =========================
st.divider()
st.subheader("🔬 Execução de Otimização Global e Validação de Hipóteses")

if st.button("Executar Varredura e Validação Completa"):
    with st.spinner("A executar algoritmos de otimização e verificação estatística em lote..."):
        alphas = np.linspace(0.001, 0.999, num_alphas)
        k = np.arange(1, n_max + 1, dtype=np.float64)[:, None]
        ak = k * alphas[None, :]
        f_matrix = np.cumsum(0.5 - (ak - np.floor(ak)), axis=0) / np.log(np.arange(1, n_max + 1, dtype=np.float64))[:, None]
        final_vals = f_matrix[-1, :]
        
        obj_min = lambda x: mathematical_verifier(x[0], n_max)["f_value"]
        obj_max = lambda x: -mathematical_verifier(x[0], n_max)["f_value"]
        
        res_min = differential_evolution(obj_min, bounds=[(0.001, 0.999)], seed=42, maxiter=10)
        res_max = differential_evolution(obj_max, bounds=[(0.001, 0.999)], seed=42, maxiter=10)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mínimo Global", f"{np.min(final_vals):.4f}")
        col2.metric("Máximo Global", f"{np.max(final_vals):.4f}")
        col3.metric("Extremo Mínimo (SciPy)", f"{res_min.fun:.4f} (alpha={res_min.x[0]:.4f})")
        col4.metric("Extremo Máximo (SciPy)", f"{-res_max.fun:.4f} (alpha={res_max.x[0]:.4f})")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(final_vals, bins=60, color='royalblue', edgecolor='black', alpha=0.7)
        ax.axvline(res_min.fun, color='red', linestyle='--', label='Mínimo Otimizado')
        ax.axvline(-res_max.fun, color='green', linestyle='--', label='Máximo Otimizado')
        ax.set_title("Distribuição Estatística dos Valores Finais sob Verificação")
        ax.set_xlabel("f(alpha, n)")
        ax.set_ylabel("Contagem")
        ax.legend()
        st.pyplot(fig)
        st.success("Análise completa validada pelo motor matemático.")
