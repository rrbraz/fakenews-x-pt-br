import os
import random

import numpy as np
import pandas as pd
import torch

from src.utils.embedding_utils import get_bert_embeddings

# --- Configurações ---
SEED = 42
EMBEDDINGS_MODEL = "neuralmind/bert-base-portuguese-cased"
EMBEDDINGS_REVISION = "94d69c95f98f7d5b2a8700c420230ae10def0baa"
INPUT_PARQUET_PATH = (
    "data/generated/dataset_expanded.parquet"
)
OUTPUT_DIR = "data/generated/"
OUTPUT_FILE = "dataset_processed.parquet"
EMBEDDINGS_CACHE_FILE = "bert_embeddings_cache.pt"


def _set_seed(seed: int) -> None:
    """Seed básico para reprodutibilidade da extração de embeddings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data():
    """Carrega, processa e salva os dados e embeddings para a avaliação LODO."""
    print("Iniciando a preparação dos dados...")
    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Carregar Dados ---
    if not os.path.exists(INPUT_PARQUET_PATH):
        print(f"Erro: Arquivo de entrada não encontrado em {INPUT_PARQUET_PATH}")
        return
    df = pd.read_parquet(INPUT_PARQUET_PATH)
    print(f"Dataset carregado com {len(df)} amostras.")

    # --- Processamento de Features ---
    print("Processando features numéricas, categóricas e LLM...")

    # Encodificar rótulos e domínios
    label_mapping = {"fake": 0, "true": 1}
    df["label_encoded"] = df["label"].map(label_mapping)
    if df["label_encoded"].isna().any():
        unknown = sorted(df.loc[df["label_encoded"].isna(), "label"].unique())
        raise ValueError(f"Rótulos desconhecidos: {unknown}")
    df["label_encoded"] = df["label_encoded"].astype(int)
    df["domain_encoded"] = pd.factorize(df["categoria"])[0]

    # Features Numéricas (metadata do usuário ou postagem).
    # Aplicamos apenas a transformação não-paramétrica (log1p) aqui. A
    # padronização (z-score) DEPENDE de mean/std do split de treino e por isso
    # é aplicada em runtime nos scripts `scripts/reexecute_with_*.py` via
    # `src.utils.feature_scaling.fit_numerical_scaler`. Rodar a z-score sobre
    # o dataset inteiro causaria leak da distribuição do test set para o
    # treino (em qualquer protocolo: LODO ou temporal).
    numerical_cols = [
        "favorite_count",
        "retweet_count",
        "reply_count",
        "followers_count",
        "friends_count",
        "statuses_count",
        "favourites_count",
    ]
    numerical_features = np.log1p(df[numerical_cols].fillna(0).values)

    # Features Categóricas básicas
    categorical_features_list = []

    # verified é sempre false, ignora
    # categorical_features_list.append(
    #     pd.get_dummies(df[["verified"]].fillna(False), drop_first=False)
    # )

    # Features LLM Categóricas — fixed vocabulary so the feature dim does
    # not drift with the LLM choice. The prompt defines polarity over the
    # three labels below; any other value (None, "desconhecido", or a
    # model-invented label) collapses into the all-zeros encoding instead
    # of opening a new column.
    POLARITY_VOCAB = ["positivo", "neutro", "negativo"]
    pol_dummies = pd.DataFrame(
        {f"llm_polaridade_{v}": (df.get("llm_polaridade") == v).astype(int)
         for v in POLARITY_VOCAB},
        index=df.index,
    )
    categorical_features_list.append(pol_dummies)

    # Features LLM Multi-hot (marcadores linguísticos) — same logic:
    # encode every entry in the prompt's whitelist, whether or not the
    # current LLM happened to emit it. Otherwise dim depends on the LLM
    # (Gemini observed 27 out of 29, Claude observed 28) and ablation
    # comparisons across LLM choices stop being apples-to-apples.
    ALLOWED_MARKERS = [
        "CAIXA_ALTA", "ESPACAMENTO_ENTRE_LETRAS", "PONTUACAO_EXCESSIVA",
        "USO_DE_RETICENCIAS", "PONTUACAO_INTERCALADA", "CALL_TO_ACTION",
        "ALERTA_URGENTE", "HIPERBOLE", "SINAL_CERTEZA", "SINAL_DUVIDA",
        "IRONIA_SARCASMO", "METAFORA", "LINK_EXTERNO", "URL_CURTA",
        "PRESENCA_DE_DATA", "APELO_A_AUTORIDADE", "CITACAO_CIENTIFICA",
        "LINGUAGEM_TECNICA", "HASHTAG", "USER_MENTION", "EMOJI",
        "EMOTICON_ASCII", "PRESENCA_DE_RISADA", "POLARIZACAO_NOS_VS_ELES",
        "PRONOME_COLETIVO", "DISCRIMINACAO", "PALAVRAO", "GIRIA", "JARGAO",
    ]
    allowed_set = set(ALLOWED_MARKERS)
    if "llm_marcadores_linguisticos" in df.columns:
        observed = (
            df["llm_marcadores_linguisticos"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
        )
        unused = sorted(allowed_set - set(observed))
        dropped = sorted(set(observed) - allowed_set)
        print(f"  - Marcadores linguísticos (multi-hot): "
              f"{len(ALLOWED_MARKERS)} (fixed vocab)")
        if unused:
            print(f"  - Marcadores do vocabulário não observados nesta rodada "
                  f"({len(unused)}): {unused}")
        if dropped:
            print(f"  - Marcadores fora do vocabulário, descartados "
                  f"({len(dropped)}): {dropped}")

        marker_features = pd.DataFrame(index=df.index)
        for marker in ALLOWED_MARKERS:
            marker_features[f"marker_{marker}"] = (
                df["llm_marcadores_linguisticos"]
                .fillna("")
                .str.contains(rf"\b{marker}\b", na=False)
                .astype(int)
            )
        categorical_features_list.append(marker_features)

    # Combinar todas as features categóricas
    categorical_features = pd.concat(categorical_features_list, axis=1)

    # Combinar todas as features
    all_features = np.hstack((numerical_features, categorical_features.values))
    df["features"] = list(all_features)
    print(f"Total de features combinadas: {all_features.shape[1]}")
    print(f"  - Features numéricas (metadata): {numerical_features.shape[1]}")
    print(f"  - Features categóricas (llm): {categorical_features.shape[1]}")

    # --- Geração e Cache de Embeddings ---
    embeddings_cache_path = os.path.join(OUTPUT_DIR, EMBEDDINGS_CACHE_FILE)
    if os.path.exists(embeddings_cache_path):
        print(f"Carregando embeddings do cache: {embeddings_cache_path}")
        embeddings = torch.load(embeddings_cache_path, weights_only=False)
    else:
        print("Gerando embeddings BERT... (Isso pode levar algum tempo)")
        texts = df["full_text"].tolist()
        embeddings = get_bert_embeddings(
            texts,
            model_name=EMBEDDINGS_MODEL,
            revision=EMBEDDINGS_REVISION,
            device=device,
        )
        print("Salvando embeddings em cache para uso futuro...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(embeddings, embeddings_cache_path)

    df["embedding"] = [emb.tolist() for emb in embeddings]

    # --- Salvar Dados Processados ---
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    # Selecionar colunas relevantes para salvar
    final_df = df[
        [
            "record_id",
            "tweet_id",
            "tweet_url",
            "screen_name",
            "full_text",
            "created_at",
            "label_encoded",
            "categoria",
            "domain_encoded",
            "features",
            "embedding",
        ]
    ]
    final_df.to_parquet(output_path, index=False)
    print(f"Dados preparados e salvos em: {output_path}")


if __name__ == "__main__":
    prepare_data()
