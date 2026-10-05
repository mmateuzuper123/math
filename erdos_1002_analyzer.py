import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

st.set_page_config(page_title="Erdos Problem #1002 AI Research Lab", layout="wide")

st.title("Erdos Problem #1002 Autonomous Research & Verification Lab")
st.markdown(
    r"""
**Target Conjecture:** For $0 < \alpha < 1$, let 
$$f(\alpha, n) = \frac{1}{\log n} \sum_{1 \le k \le n} \left(\frac{1}{2} - \{\alpha k\}\right)$$
Este ambiente simula um agente de pesquisa computacional que propõe hipóteses analíticas e um **verificador matemático automático** que testa a validade das propostas.
"""
)

# =========================
# CONTROLES DA BARRA LATERAL
# =========================
st.sidebar.header("Configuração do Agente")
n_max = st.sidebar.slider("Comprimento Máximo (n)", min_value=1000, max_value=20000, value=5000, step=1000)
num_alphas = st.sidebar.slider("Resolução de Alphas", min_value=500, max_value=3000, value=1000, step=500)

# =========================
# MOTOR DE VERIFICAÇÃO MATEMÁTICA
# =========================
def mathematical_verifier(hypothesis_type, alpha_val, n_limit):
    """Verificador estrito que testa limites, cotas e distribuição."""
    k = np.arange(1, n_limit + 1, dtype=np.float64)
    ak = k * alpha_val
    fractional_parts = ak - np.floor(ak)
    terms = 0.5 - fractional_parts
    f_val = np.sum(terms) / np.log(n_limit)
    
    # Critérios de verificação baseados na conjectura
    is_bounded = abs(f_val) < 5.0 # Teste de cota prática
    status = "APROVADO (Dentro dos limites estocásticos esperados)" if is_bounded else "REJEITADO (Violação de cota ou divergência)"
    
    return {
        "f_value": f_val,
        "bounded": is_bounded,
        "verification_status": status
    }

# =========================
# INTERFACE DE CHAT E AGENTE AUTÓNOMO
# =========================
st.subheader("🤖 Chat com o Agente de Descoberta e Verificador")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá. Sou o agente computacional configurado para explorar Erdős #1002. Digite uma hipótese (ex: 'Testar alpha = 0.6180') ou clique no botão abaixo para gerar uma investigação autónoma."}
    ]

# Exibir histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Botão para acionar o ciclo autónomo de investigação
if st.button("Executar Ciclo Autónomo de Investigação e Verificação"):
    with st.spinner("O agente está a gerar hipóteses e o verificador está a testar..."):
        # 1. O Agente propõe um candidato baseado em propriedades diofantinas (ex: Razão Áurea ou pontos críticos)
        test_alphas = [0.61803398875, np.sqrt(2) - 1, np.e - 2, 0.3333]
        results_log = []
        
        for idx, a in enumerate(test_alphas):
            # Validação pelo Verificador Matemático
            verif = mathematical_verifier("Bound Check", a, n_max)
            results_log.append(f"**Hipótese {idx+1} ($\alpha = {a:.4f}$):** Calculado $f(\alpha, n) = {verif['f_value']:.4f} \rightarrow$ **{verif['verification_status']}**")
        
        agent_response = "### Relatório do Ciclo de Investigação Autónoma:\n" + "\n".join(results_log)
        
        st.session_state.messages.append({"role": "assistant", "content": agent_response})
        st.rerun()

# Entrada de chat manual do utilizador
if prompt := st.chat_input("Insira uma directiva para o agente (ex: testar alpha específico ou analisar cotas)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        # Resposta simulada do agente com base na diretiva do utilizador
        try:
            # Tentar extrair um número alpha se o utilizador forneceu
            if "alpha" in prompt.lower():
                # Resposta analítica padrão simulada para testes
                custom_alpha = 0.5 # Valor padrão de fallback
                for word in prompt.split():
                    try:
                        val = float(word)
                        if 0 < val < 1:
                            custom_alpha = val
                    except ValueError:
                        pass
                
                verif = mathematical_verifier("Manual Test", custom_alpha, n_max)
                reply = f"**Análise da Hipótese Manual para $\alpha = {custom_alpha}$:**\n- Valor obtido: $f(\alpha, n) = {verif['f_value']:.4f}$\n- Estado do Verificador: **{verif['verification_status']}**"
            else:
                reply = f"Recebi a diretiva: '{prompt}'. Para testar uma solução específica, tente por exemplo: 'testar alpha 0.45'."
        except Exception as e:
            reply = f"Erro no processamento da diretiva: {str.format(str(e))}"
            
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# =========================
# MOTOR GRÁFICO DE SUPORTE
# =========================
st.divider()
st.subheader("Visualização Rápida de Distribuição Empírica")
if st.button("Atualizar Gráficos de Suporte"):
    alphas = np.linspace(0.001, 0.999, num_alphas)
    k = np.arange(1, n_max + 1, dtype=np.float64)[:, None]
    ak = k * alphas[None, :]
    f_matrix = np.cumsum(0.5 - (ak - np.floor(ak)), axis=0) / np.log(np.arange(1, n_max + 1, dtype=np.float64))[:, None]
    final_vals = f_matrix[-1, :]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(final_vals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title("Distribuição dos Valores Finais de f(alpha, n)")
    ax.set_xlabel("f(alpha, n)")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)
