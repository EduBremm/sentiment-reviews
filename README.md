# 📱 App Review Intelligence — Mineração de Sentimentos em Reviews de Aplicativos

Plataforma de análise automatizada de reviews da Google Play Store que classifica sentimento (POSITIVO / NEUTRO / NEGATIVO), identifica clusters de reclamações e apresenta indicadores em um dashboard.

Projeto da disciplina de Mineração de dados — Centro Universitário FAG.
Autor: **Eduardo Luis Bremm**.

---

## 🎯 Problema de negócio

Aplicações recebem milhares de avaliações. Equipes de produto não conseguem lê-las manualmente para identificar rapidamente a percepção dos usuários. Este projeto transforma textos não estruturados em indicadores acionáveis de satisfação.

---

## 🧭 CRISP-DM

| Fase | Onde no repositório |
|---|---|
| 1. Business Understanding | Este README + `notebooks/01_eda.ipynb` |
| 2. Data Understanding | `notebooks/01_eda.ipynb` |
| 3. Data Preparation | `src/data/preprocess.py` |
| 4. Modeling | `src/models/train.py` + `notebooks/02_modeling.ipynb` |
| 5. Evaluation | `notebooks/02_modeling.ipynb` (matriz de confusão, F1, ROC-AUC) |
| 6. Deployment | `src/api/main.py` (FastAPI) + `dashboard/app.py` (Streamlit) |

---

## 🏗️ Arquitetura

```
┌──────────────────┐
│  Coleta          │  google-play-scraper
│  (src/data)      │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  PostgreSQL      │  reviews, predictions, clusters
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Pipeline ML     │  scikit-learn (NB, LogReg, SVM)
│  (src/models)    │  + K-Means para clusters de reclamação
└────────┬─────────┘
         ▼
┌──────────────────┐  ┌──────────────────┐
│  API FastAPI     │  │  Dashboard       │
│  :8000           │  │  Streamlit :8501 │
└──────────────────┘  └──────────────────┘
```

---

## 🚀 Como rodar (Docker — recomendado)

```bash
# 1. Clonar
git clone <seu-repo> sentiment-reviews
cd sentiment-reviews

# 2. Configurar variáveis de ambiente
cp .env.example .env

# 3. Subir tudo (Postgres + API + Dashboard)
docker compose up -d --build

# 4. Coletar reviews (roda dentro do container da API)
docker compose exec api python -m src.data.collect --app-id com.whatsapp --count 2000

# 5. Treinar modelos
docker compose exec api python -m src.models.train

# 6. Rodar clusterização de reclamações
docker compose exec api python -m src.models.clustering

# Acessos:
# - API:       http://localhost:8000/docs
# - Dashboard: http://localhost:8501
```

## 🖥️ Como rodar (sem Docker)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt

# Postgres local rodando na porta 5432 e banco criado
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/reviews

python -m src.data.collect --app-id com.whatsapp --count 2000
python -m src.models.train
python -m src.models.clustering

uvicorn src.api.main:app --reload --port 8000
streamlit run dashboard/app.py
```

---

## 📦 Estrutura do projeto

```
sentiment-reviews/
├── src/
│   ├── config.py                 # Configurações centralizadas
│   ├── data/
│   │   ├── collect.py            # Coleta via google-play-scraper
│   │   └── preprocess.py         # Limpeza e normalização de texto
│   ├── models/
│   │   ├── train.py              # Treina NB, LogReg, SVM e escolhe o campeão
│   │   ├── predict.py            # Carrega modelo e prediz sentimento
│   │   └── clustering.py         # K-Means em reviews negativos
│   ├── api/
│   │   ├── main.py               # FastAPI (predict, stats, reviews)
│   │   └── schemas.py            # Modelos Pydantic
│   └── db/
│       ├── database.py           # Engine SQLAlchemy
│       └── models.py             # Tabelas ORM
├── dashboard/
│   └── app.py                    # Streamlit
├── notebooks/
│   ├── 01_eda.ipynb              # Análise Exploratória
│   └── 02_modeling.ipynb         # Comparação de modelos e avaliação
├── tests/
│   ├── test_preprocess.py
│   ├── test_api.py
│   └── test_models.py
├── scripts/
│   └── init_db.sql
├── .github/workflows/ci.yml      # CI: lint + pytest
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📊 Modelos comparados

| Modelo | Vetorização | Uso |
|---|---|---|
| Naive Bayes (Multinomial) | TF-IDF | Baseline |
| Regressão Logística | TF-IDF | Comparação |
| SVM (LinearSVC) | TF-IDF | Comparação |

O modelo com melhor **F1-Score macro** é salvo como `models/best_model.pkl` e usado pela API.

### Métricas usadas
- **Classificação:** Accuracy, Precision, Recall, F1-Score (macro), ROC-AUC (OvR).
- **Clustering:** Silhouette Score, Davies-Bouldin.

---

## 🔬 Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| GET  | `/health` | Health check |
| POST | `/predict` | Recebe um texto e retorna sentimento + confiança |
| POST | `/predict/batch` | Lista de textos |
| GET  | `/reviews/stats` | Contagens agregadas por sentimento e versão |
| GET  | `/reviews/timeline` | Série temporal de sentimento |
| GET  | `/reviews/clusters` | Clusters de reclamação com termos representativos |

Docs interativas em `http://localhost:8000/docs`.

---

## ✅ Reprodutibilidade

- Todo o pipeline reproduzível com `docker compose up`
- Testes: `pytest`
- CI no GitHub Actions em `.github/workflows/ci.yml`
- Seeds fixadas em `src/config.py` para reprodutibilidade dos experimentos

---

## 📄 Licença

Uso acadêmico — Centro Universitário FAG, 2026.
