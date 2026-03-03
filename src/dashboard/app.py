"""Dashboard Streamlit."""
import streamlit as st

st.set_page_config(page_title="Fuel Predictor", layout="wide")
st.title("Sistema de Previsao de Precos de Combustiveis")
st.caption("Dados: ANP | Projeto TCC")
st.info("Dashboard em construcao. Disponivel na Fase 3 (Mes 11).")

with st.sidebar:
    st.header("Filtros")
    estado = st.selectbox("Estado", ["SP", "RJ", "MG", "PR", "RS"])
    produto = st.selectbox("Produto", ["GASOLINA COMUM", "ETANOL", "DIESEL S10"])
