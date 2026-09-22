#!/usr/bin/env bash
# Pipeline completo: coletar → treinar → clusterizar
set -euo pipefail

APP_ID="${1:-com.whatsapp}"
COUNT="${2:-2000}"

echo "==> 1/3 Coletando $COUNT reviews de $APP_ID"
python -m src.data.collect --app-id "$APP_ID" --count "$COUNT"

echo "==> 2/3 Treinando modelos"
python -m src.models.train

echo "==> 3/3 Rodando clusterização"
python -m src.models.clustering

echo "✅ Pipeline concluído."
