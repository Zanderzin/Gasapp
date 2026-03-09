"""Dashboard GasApp — Precos de Combustiveis no DF."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.etl.loader import load_parquet

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GasApp DF",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    
    .stMetric { background: #0f1117; border: 1px solid #2d2d2d; border-radius: 8px; padding: 16px; }
    .stMetric label { color: #888 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
    .stMetric [data-testid="metric-container"] { color: #fff; }
    
    h1 { font-family: 'IBM Plex Mono', monospace !important; font-size: 28px !important; color: #f5c518 !important; }
    h2 { font-family: 'IBM Plex Mono', monospace !important; font-size: 18px !important; color: #fff !important; }
    h3 { color: #aaa !important; font-weight: 300 !important; font-size: 14px !important; }
    
    .sidebar .sidebar-content { background: #0a0a0a; }
    
    div[data-testid="stSidebar"] { background: #0a0a0a; border-right: 1px solid #1e1e1e; }
    
    .barato { color: #22c55e; font-weight: 600; }
    .caro { color: #ef4444; font-weight: 600; }
    
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-family: 'IBM Plex Mono', monospace; }
    .tag-green { background: #14532d; color: #22c55e; }
    .tag-red { background: #450a0a; color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def carregar_dados():
    df = load_parquet()
    df["semana"] = df["data_coleta"].dt.to_period("W").dt.start_time
    return df


@st.cache_data(ttl=3600)
def agregar_regiao(df):
    return (
        df.groupby(["regiao", "produto", "semana"])
        .agg(preco_medio=("preco_revenda", "mean"), n_postos=("cnpj", "nunique"))
        .reset_index()
    )


def carregar_modelo(produto: str):
    path = Path("models") / f"xgb_{produto.lower().replace(' ', '_')}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
df = carregar_dados()
df_reg = agregar_regiao(df)

PRODUTOS_LIQUIDOS = ["GASOLINA COMUM", "GASOLINA ADITIVADA", "ETANOL", "DIESEL S10", "DIESEL S500"]

with st.sidebar:
    st.markdown("# ⛽ GasApp DF")
    st.markdown("---")
    pagina = st.radio("Navegacao", ["Comparador", "Previsao", "Analise"], label_visibility="collapsed")
    st.markdown("---")
    produto = st.selectbox("Combustivel", PRODUTOS_LIQUIDOS)
    regioes = sorted(df["regiao"].unique())
    regiao = st.selectbox("Regiao", regioes, index=regioes.index("TAGUATINGA") if "TAGUATINGA" in regioes else 0)
    st.markdown("---")
    ultima = df["data_coleta"].max()
    st.caption(f"Dados ate: **{ultima.strftime('%d/%m/%Y')}**")
    st.caption(f"Postos: **{df['cnpj'].nunique()}**")
    st.caption(f"Fonte: ANP")


# ── Pagina 1: Comparador ──────────────────────────────────────────────────────
if pagina == "Comparador":
    st.markdown(f"# Comparador de Precos")
    st.markdown(f"### {produto} · {regiao}")
    st.markdown("---")

    # Ultimas 4 semanas
    df_prod = df[(df["produto"] == produto)].copy()
    ultima_semana = df_prod["semana"].max()
    df_recente = df_prod[df_prod["semana"] >= ultima_semana - pd.Timedelta(weeks=3)]

    # Metricas gerais
    preco_atual = df_recente[df_recente["regiao"] == regiao]["preco_revenda"].mean()
    preco_df = df_recente["preco_revenda"].mean()
    preco_min = df_recente["preco_revenda"].min()
    preco_max = df_recente["preco_revenda"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preco medio — regiao", f"R$ {preco_atual:.2f}", f"{preco_atual - preco_df:+.2f} vs DF")
    c2.metric("Media do DF", f"R$ {preco_df:.2f}")
    c3.metric("Mais barato do DF", f"R$ {preco_min:.2f}")
    c4.metric("Mais caro do DF", f"R$ {preco_max:.2f}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("## Ranking de postos")
        df_reg_prod = df_recente[df_recente["regiao"] == regiao]
        ranking = (
            df_reg_prod.groupby(["cnpj", "nome_fantasia", "bandeira"])["preco_revenda"]
            .mean().reset_index().sort_values("preco_revenda")
        )
        ranking["posicao"] = range(1, len(ranking) + 1)
        ranking["preco_fmt"] = ranking["preco_revenda"].apply(lambda x: f"R$ {x:.2f}")

        media_reg = ranking["preco_revenda"].mean()
        ranking["vs_media"] = ranking["preco_revenda"] - media_reg
        ranking["tag"] = ranking["vs_media"].apply(
            lambda x: "🟢 barato" if x < -0.02 else ("🔴 caro" if x > 0.02 else "⚪ medio")
        )

        for _, row in ranking.head(10).iterrows():
            st.markdown(f"**{int(row['posicao'])}. {row['nome_fantasia'][:30]}** &nbsp; {row['preco_fmt']} &nbsp; {row['tag']}")

    with col2:
        st.markdown("## Comparativo entre regioes")
        preco_por_regiao = (
            df_recente.groupby("regiao")["preco_revenda"].mean()
            .sort_values().reset_index()
        )
        preco_por_regiao["cor"] = preco_por_regiao["regiao"].apply(
            lambda r: "#22c55e" if r == regiao else "#3b82f6"
        )
        fig = go.Figure(go.Bar(
            x=preco_por_regiao["preco_revenda"],
            y=preco_por_regiao["regiao"],
            orientation="h",
            marker_color=preco_por_regiao["cor"],
            text=preco_por_regiao["preco_revenda"].apply(lambda x: f"R$ {x:.2f}"),
            textposition="outside",
        ))
        fig.update_layout(
            height=500, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#fff", size=11),
            xaxis=dict(showgrid=False, visible=False, range=[0, preco_por_regiao["preco_revenda"].max() * 1.15]),
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=10, r=80, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Pagina 2: Previsao ────────────────────────────────────────────────────────
elif pagina == "Previsao":
    st.markdown("# Previsao de Preco")
    st.markdown(f"### {produto} · {regiao} · proxima semana")
    st.markdown("---")

    modelo_data = carregar_modelo(produto)

    if modelo_data is None:
        st.warning(f"Modelo para {produto} nao encontrado. Rode: py scripts/train_models.py --salvar")
    else:
        from xgboost import XGBRegressor
        import sys
        sys.path.insert(0, '.')
        from scripts.train_models import agregar_por_regiao, build_features

        df_agg = agregar_por_regiao(df)
        df_feat = build_features(df_agg)
        df_reg_prod = df_feat[(df_feat["regiao"] == regiao) & (df_feat["produto"] == produto)]

        if len(df_reg_prod) == 0:
            st.error(f"Sem dados suficientes para {regiao} / {produto}")
        else:
            ultima_linha = df_reg_prod.sort_values("data_coleta").iloc[-1]
            feat_cols = modelo_data["feature_cols"]
            X = ultima_linha[feat_cols].values.reshape(1, -1)
            delta_pred = modelo_data["model"].predict(X)[0]
            preco_atual = ultima_linha["preco_revenda"]
            preco_prev = preco_atual + delta_pred
            variacao = delta_pred

            c1, c2, c3 = st.columns(3)
            c1.metric("Preco atual", f"R$ {preco_atual:.2f}")
            c2.metric("Previsao proxima semana", f"R$ {preco_prev:.2f}", f"{variacao:+.2f}")
            c3.metric("Tendencia", "📈 Alta" if variacao > 0.01 else ("📉 Queda" if variacao < -0.01 else "➡️ Estavel"))

            st.markdown("---")
            st.markdown("## Historico + Previsao")

            hist = df_reg[(df_reg["regiao"] == regiao) & (df_reg["produto"] == produto)].sort_values("semana").tail(20)
            prox_semana = hist["semana"].max() + pd.Timedelta(weeks=1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["semana"], y=hist["preco_medio"],
                mode="lines+markers", name="Historico",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=[hist["semana"].iloc[-1], prox_semana],
                y=[preco_atual, preco_prev],
                mode="lines+markers", name="Previsao",
                line=dict(color="#f5c518", width=2, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
            ))
            fig.update_layout(
                height=350, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font=dict(color="#fff"),
                legend=dict(bgcolor="#0f1117"),
                xaxis=dict(gridcolor="#1e1e1e"),
                yaxis=dict(gridcolor="#1e1e1e", tickprefix="R$ "),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.caption("⚠️ Previsao baseada em dados historicos da ANP. Nao garante precisao futura.")


# ── Pagina 3: Analise ─────────────────────────────────────────────────────────
elif pagina == "Analise":
    st.markdown("# Analise Historica")
    st.markdown(f"### {produto}")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## Evolucao de preco — regioes selecionadas")
        regioes_sel = st.multiselect(
            "Selecione regioes para comparar",
            regioes,
            default=["TAGUATINGA", "ASA NORTE", "CEILANDIA", "AGUAS CLARAS"] if all(r in regioes for r in ["TAGUATINGA", "ASA NORTE", "CEILANDIA", "AGUAS CLARAS"]) else regioes[:4],
        )
        df_plot = df_reg[(df_reg["produto"] == produto) & (df_reg["regiao"].isin(regioes_sel))]
        fig = px.line(
            df_plot, x="semana", y="preco_medio", color="regiao",
            template="plotly_dark",
            labels={"preco_medio": "Preco medio (R$)", "semana": "Semana", "regiao": "Regiao"},
        )
        fig.update_layout(
            height=380, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            legend=dict(bgcolor="#0f1117", orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("## Ranking atual")
        ultima_sem = df_reg["semana"].max()
        df_rank = (
            df_reg[(df_reg["produto"] == produto) & (df_reg["semana"] >= ultima_sem - pd.Timedelta(weeks=3))]
            .groupby("regiao")["preco_medio"].mean()
            .sort_values().reset_index()
        )
        media_geral = df_rank["preco_medio"].mean()
        for i, row in df_rank.iterrows():
            diff = row["preco_medio"] - media_geral
            cor = "#22c55e" if diff < -0.02 else ("#ef4444" if diff > 0.02 else "#888")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e1e1e'>"
                f"<span style='color:#ccc;font-size:13px'>{row['regiao']}</span>"
                f"<span style='color:{cor};font-family:monospace;font-size:13px'>R$ {row['preco_medio']:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("## Sazonalidade mensal")
    df_saz = df[(df["produto"] == produto)].copy()
    df_saz["mes_nome"] = df_saz["data_coleta"].dt.month
    preco_mes = df_saz.groupby("mes_nome")["preco_revenda"].mean().reset_index()
    meses = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
    preco_mes["mes_nome"] = preco_mes["mes_nome"].map(meses)
    fig2 = px.bar(preco_mes, x="mes_nome", y="preco_revenda", template="plotly_dark",
                  labels={"preco_revenda": "Preco medio (R$)", "mes_nome": "Mes"},
                  color="preco_revenda", color_continuous_scale="YlOrRd")
    fig2.update_layout(height=280, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                       showlegend=False, coloraxis_showscale=False,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
