"""Dashboard Streamlit — App Review Intelligence."""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

# Garante que /app (raiz do projeto) esteja no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from src.data.preprocess import preprocess

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="App Review Intelligence",
    page_icon="📱",
    layout="wide",
)

SENTIMENT_COLORS = {"POSITIVO": "#22c55e", "NEUTRO": "#94a3b8", "NEGATIVO": "#ef4444"}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@st.cache_data(ttl=60)
def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Erro chamando API {path}: {exc}")
        return None


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("📱 App Review Intelligence")
st.sidebar.caption("FAG • Eduardo Bremm")

page = st.sidebar.radio(
    "Navegação",
    ["🏠 Visão geral", "📈 Evolução temporal", "📦 Por versão", "🧩 Clusters de reclamação", "🔮 Classificar review"],
)

health = api_get("/health")
if health:
    ok = "🟢" if health["status"] == "ok" else "🟡"
    st.sidebar.markdown(f"**Status:** {ok} `{health['status']}`")
    st.sidebar.markdown(f"Modelo: {'✅' if health['model_loaded'] else '❌'}")
    st.sidebar.markdown(f"BD: {'✅' if health['db_ok'] else '❌'}")


# ------------------------------------------------------------
# Página: Visão geral
# ------------------------------------------------------------
if page == "🏠 Visão geral":
    st.title("Visão geral")

    stats = api_get("/reviews/stats") or []
    total = sum(s["count"] for s in stats)

    counts = {s["sentiment"]: s["count"] for s in stats}
    pos, neu, neg = counts.get("POSITIVO", 0), counts.get("NEUTRO", 0), counts.get("NEGATIVO", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de reviews", f"{total:,}".replace(",", "."))
    c2.metric("😀 Positivos", f"{pos:,}".replace(",", "."), f"{(pos/total*100 if total else 0):.1f}%")
    c3.metric("😐 Neutros", f"{neu:,}".replace(",", "."), f"{(neu/total*100 if total else 0):.1f}%")
    c4.metric("😡 Negativos", f"{neg:,}".replace(",", "."), f"{(neg/total*100 if total else 0):.1f}%")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Distribuição de sentimentos")
        if stats:
            df = pd.DataFrame(stats)
            fig = px.pie(
                df, values="count", names="sentiment",
                color="sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Nuvem de palavras — reviews NEGATIVOS")
        clusters = api_get("/reviews/clusters") or []
        if clusters:
            all_terms = []
            for c in clusters:
                all_terms.extend(c["top_terms"])
            all_samples = " ".join(
                preprocess(s) for c in clusters for s in c["sample_reviews"]
            )
            if all_samples.strip():
                wc = WordCloud(width=600, height=400, background_color="white", colormap="Reds").generate(all_samples)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Sem texto suficiente para nuvem.")
        else:
            st.info("Rode a clusterização (`python -m src.models.clustering`) para ver esta visualização.")


# ------------------------------------------------------------
# Página: Evolução temporal
# ------------------------------------------------------------
elif page == "📈 Evolução temporal":
    st.title("Evolução temporal do sentimento")

    gran = st.selectbox("Granularidade", ["day", "week", "month"], index=1)
    data = api_get("/reviews/timeline", {"granularity": gran}) or []
    if not data:
        st.info("Sem dados. Colete reviews antes.")
    else:
        df = pd.DataFrame(data)
        df_long = df.melt(id_vars="date", value_vars=["positive", "neutral", "negative"],
                          var_name="sentimento", value_name="quantidade")
        label_map = {"positive": "POSITIVO", "neutral": "NEUTRO", "negative": "NEGATIVO"}
        df_long["sentimento"] = df_long["sentimento"].map(label_map)
        fig = px.line(
            df_long, x="date", y="quantidade", color="sentimento",
            color_discrete_map=SENTIMENT_COLORS, markers=True,
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Dados brutos")
        st.dataframe(df, use_container_width=True)


# ------------------------------------------------------------
# Página: Por versão
# ------------------------------------------------------------
elif page == "📦 Por versão":
    st.title("Sentimento por versão do app")

    data = api_get("/reviews/by-version") or []
    if not data:
        st.info("Sem dados por versão.")
    else:
        df = pd.DataFrame(data)
        df_long = df.melt(
            id_vars="app_version", value_vars=["positive", "neutral", "negative"],
            var_name="sentimento", value_name="quantidade",
        )
        label_map = {"positive": "POSITIVO", "neutral": "NEUTRO", "negative": "NEGATIVO"}
        df_long["sentimento"] = df_long["sentimento"].map(label_map)

        fig = px.bar(
            df_long, x="app_version", y="quantidade", color="sentimento",
            color_discrete_map=SENTIMENT_COLORS, barmode="stack",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("% de reviews negativos por versão")
        df["pct_neg"] = (df["negative"] / df["total"] * 100).round(1)
        fig2 = px.bar(
            df.sort_values("pct_neg", ascending=False), x="app_version", y="pct_neg",
            color="pct_neg", color_continuous_scale="Reds",
        )
        fig2.update_layout(yaxis_title="% negativo")
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df, use_container_width=True)


# ------------------------------------------------------------
# Página: Clusters
# ------------------------------------------------------------
elif page == "🧩 Clusters de reclamação":
    st.title("Clusters de reclamação (K-Means)")
    st.caption("Temas recorrentes nos reviews negativos.")

    clusters = api_get("/reviews/clusters") or []
    if not clusters:
        st.info("Rode `python -m src.models.clustering` primeiro.")
    else:
        for c in clusters:
            with st.expander(f"🏷️ Cluster {c['cluster_id']} • {c['size']} reviews • termos: {', '.join(c['top_terms'][:5])}"):
                st.markdown("**Top termos:** " + ", ".join(f"`{t}`" for t in c["top_terms"]))
                st.markdown("**Exemplos:**")
                for s in c["sample_reviews"]:
                    st.markdown(f"> {s}")


# ------------------------------------------------------------
# Página: Classificar review
# ------------------------------------------------------------
elif page == "🔮 Classificar review":
    st.title("Classificar um review")
    st.caption("Cole ou escreva um texto para prever o sentimento em tempo real.")

    txt = st.text_area("Texto do review", height=150,
                       placeholder="Ex: O app ficou muito lento depois da última atualização...")
    if st.button("Classificar", type="primary", disabled=not txt.strip()):
        try:
            res = api_post("/predict", {"text": txt})
            emoji = {"POSITIVO": "😀", "NEUTRO": "😐", "NEGATIVO": "😡"}.get(res["sentiment"], "❓")
            st.success(f"{emoji} **{res['sentiment']}** — confiança: {res.get('confidence') or 0:.1%}")
            probs = res.get("probabilities") or {}
            if probs:
                df = pd.DataFrame([{"sentimento": k, "prob": v} for k, v in probs.items()])
                fig = px.bar(df, x="sentimento", y="prob", color="sentimento",
                             color_discrete_map=SENTIMENT_COLORS)
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Erro na predição: {exc}")
