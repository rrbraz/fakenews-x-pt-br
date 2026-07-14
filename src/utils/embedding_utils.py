"""Text feature extraction used by the released benchmark."""
from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def extract_tfidf_features(
    train_texts,
    val_texts,
    test_texts,
    n_gram_range=(1, 2),
    max_features=None,
):
    """Fit TF-IDF on training text and transform validation and test text."""
    vectorizer = TfidfVectorizer(
        ngram_range=n_gram_range,
        max_features=max_features,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_texts)
    val_features = vectorizer.transform(val_texts)
    test_features = vectorizer.transform(test_texts)
    return train_features, val_features, test_features, vectorizer


def select_features(train_features, train_labels, val_features, test_features, k=1000):
    """Select the training-fitted top-k chi-squared TF-IDF features."""
    selector = SelectKBest(chi2, k=k)
    train_selected = selector.fit_transform(train_features, train_labels)
    val_selected = selector.transform(val_features)
    test_selected = selector.transform(test_features)
    return train_selected, val_selected, test_selected, selector


def get_bert_embeddings(
    texts,
    model_name="neuralmind/bert-base-portuguese-cased",
    revision=None,
    max_length=128,
    device=None,
):
    """Generate frozen BERT `[CLS]` embeddings with a per-text local cache."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = os.environ.get(
        "BERT_CACHE_DIR",
        f'data/generated/embedding_cache/{model_name.replace("/", "_")}',
    )
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(
        model_name, revision=revision, do_lower_case=False
    )
    model = BertModel.from_pretrained(model_name, revision=revision).to(device)
    model.eval()

    embeddings = []
    for text in tqdm(texts, desc="Extracting BERT embeddings"):
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{text_hash}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as file:
                embedding = pickle.load(file)
        else:
            encoded = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            with torch.no_grad():
                embedding = (
                    model(**encoded).last_hidden_state[:, 0, :].cpu().numpy().squeeze(0)
                )
            with open(cache_path, "wb") as file:
                pickle.dump(embedding, file)
        embeddings.append(embedding)
    return np.vstack(embeddings)
