import hashlib
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

SYSTEM_PROMPT = """Você é um “Assistente de Enriquecimento de Contexto”.  
Sua tarefa é receber o texto bruto de um tweet (incluindo eventuais RTs, citações ou respostas) e produzir **apenas informações contextualizadoras** que possam auxiliar modelos de detecção de fake news.  

## Instruções

1. Analise o tweet fornecido entre os marcadores <<<TWEET>>> e <<<FIM_TWEET>>>.  
2. Extraia e apresente, em **português**, os seguintes campos:
  
- **polaridade**: escolha entre {“neutro”, “positivo”, “negativo”}.
- **marcadores_linguisticos**: array com os marcadores que aparecem no texto. Escolha entre ["CAIXA_ALTA", "ESPACAMENTO_ENTRE_LETRAS", "PONTUACAO_EXCESSIVA", "USO_DE_RETICENCIAS", "PONTUACAO_INTERCALADA", "CALL_TO_ACTION", "ALERTA_URGENTE", "HIPERBOLE", "SINAL_CERTEZA", "SINAL_DUVIDA", "IRONIA_SARCASMO", "METAFORA", "LINK_EXTERNO", "URL_CURTA", "PRESENCA_DE_DATA", "APELO_A_AUTORIDADE", "CITACAO_CIENTIFICA", "LINGUAGEM_TECNICA", "HASHTAG", "USER_MENTION", "EMOJI", "EMOTICON_ASCII", "PRESENCA_DE_RISADA", "POLARIZACAO_NOS_VS_ELES", "PRONOME_COLETIVO", "DISCRIMINACAO", "PALAVRAO", "GIRIA", "JARGAO"].  
- **raciocinio**: elabore um raciocínio sucinto sobre os

3. Selecione apenas os rótulos dentre os listados, não crie nenhum novo.
4. **Formato de saída**: JSON válido. Se ocorrer erro de sintaxe, reenvie até ser válido. 

## Exemplo de uso

Entrada:
<<<TWEET>>>
A nova vacina da OMS altera seu DNA! Veja o link: https://exemplo.com #Alerta
<<<FIM_TWEET>>>

Saída esperada:
{
  "polaridade": "negativo",
  "marcadores_linguisticos": [
    "ALERTA_URGENTE",
    "LINK_EXTERNO",
    "HASHTAG",
    "SINAL_CERTEZA"
  ],
  "raciocinio": "O tweet expressa desconfiança e medo em relação à vacina, apresentando uma afirmação categórica e alarmista ('altera seu DNA'), acompanhada de uma hashtag de alerta e um link externo, sugerindo tentativa de gerar pânico ou desinformação."
}
"""


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV or Parquet file.

    Args:
        file_path: Path to the CSV or Parquet file

    Returns:
        DataFrame containing the dataset
    """
    print(f"Loading dataset from {file_path}")
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    return df


def create_prompt(tweet_text: str) -> str:
    """
    Create a prompt for the LLM to analyze the tweet.

    Args:
        tweet_text: The text of the tweet to analyze

    Returns:
        A formatted prompt for the LLM
    """
    return f"""Entrada:
<<<TWEET>>>
{tweet_text}
<<<FIM_TWEET>>>"""


def call_openrouter_api(
    prompt: str,
    model: str = "openai/gpt-4.1",
    api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Dict:
    """
    Call the OpenRouter API with the given prompt.

    Args:
        prompt: The prompt to send to the LLM
        model: The model to use
        api_key: OpenRouter API key (default: None, will use environment variable)
        max_retries: Maximum number of retries on failure
        retry_delay: Delay between retries in seconds
        temperature: Temperature parameter for the LLM
        max_tokens: Maximum number of tokens in the response

    Returns:
        The API response as a dictionary
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set in environment. Please provide it as an argument."
            )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/user/repo",  # Replace with your actual URL
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "structured_outputs": True,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=600,
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"Error calling OpenRouter API: {e}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
                # Increase delay for next retry
                retry_delay *= 2
            else:
                print(
                    f"Failed to call OpenRouter API after {max_retries} attempts: {e}"
                )
                raise


def _fetch_one(text: str, model: str, api_key: str, cache_dir: str) -> Dict:
    """Resolve one text → cached or fresh API response, persisting on miss.

    Pure per-text function: safe to invoke in a ThreadPoolExecutor because the
    only shared state is the cache directory and we write atomically (one
    file per text hash).
    """
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{text_hash}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    prompt = create_prompt(text)
    response = call_openrouter_api(prompt, model=model, api_key=api_key)
    # Write to a temp file then rename — no half-written pickles if interrupted.
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(response, f)
    os.replace(tmp_path, cache_path)
    return response


def get_llm_analysis(
    texts: List[str],
    model: str = "openai/gpt-4.1",
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    workers: int = 1,
) -> List[Dict]:
    """Resolve LLM analysis for a list of texts, parallelized over ``workers``.

    Cache layout (one ``.pkl`` per text hash) is unchanged from the original
    serial version, so partial runs can be resumed cheaply: any text whose
    cache file already exists is served from disk and skipped from the API.

    When ``workers > 1``, missing texts are dispatched concurrently via a
    ``ThreadPoolExecutor``. The OpenRouter API is I/O-bound, so threads
    suffice; the GIL releases during the ``requests.post`` socket wait.

    Ordering is preserved: results come back in the same order as ``texts``,
    regardless of completion order in the pool.
    """
    if cache_dir is None:
        cache_dir = f"data/llm_cache/{model.replace('/', '_')}"
    os.makedirs(cache_dir, exist_ok=True)

    # First pass: serve hits from cache, queue misses for the pool.
    all_responses: List[Optional[Dict]] = [None] * len(texts)
    todo: List[int] = []
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{text_hash}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                all_responses[i] = pickle.load(f)
        else:
            todo.append(i)

    print(f"Cache hits: {len(texts) - len(todo)} / {len(texts)}")
    if not todo:
        return all_responses
    print(f"Fetching {len(todo)} texts with {workers} worker(s)...")

    if workers <= 1:
        for i in tqdm(todo, desc="Processing with LLM"):
            try:
                all_responses[i] = _fetch_one(texts[i], model, api_key, cache_dir)
            except Exception as e:
                print(f"Error processing index {i}: {e}")
        return all_responses

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, texts[i], model, api_key, cache_dir): i
                   for i in todo}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"Processing with LLM ({workers}x)"):
            i = futures[fut]
            try:
                all_responses[i] = fut.result()
            except Exception as e:
                print(f"Error processing index {i}: {e}")

    return all_responses


def extract_llm_content(response: Dict) -> str:
    """Extract the content string from an OpenRouter chat-completion response.

    Some providers (notably Anthropic routed via Bedrock) wrap JSON output in
    ```json ... ``` fences even when ``response_format=json_object`` is
    requested. We strip those fences here so downstream parsers
    (``02_expand_llm.py``) can call ``json.loads`` directly without
    knowing which provider produced the response.
    """
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error extracting content from response: {e}")
        return ""
    if not isinstance(content, str):
        return ""
    s = content.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present.
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    return s


def enrich_dataset(
    df: pd.DataFrame,
    model: str = "openai/gpt-4.1",
    api_key: Optional[str] = None,
    sample_size: Optional[int] = None,
    output_path: Optional[str] = None,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Enrich the dataset with LLM analysis.

    Args:
        df: DataFrame containing the dataset
        model: LLM model to use
        api_key: OpenRouter API key
        sample_size: Number of rows to process (for testing)
        output_path: Path to save the enriched dataset
        workers: Number of concurrent API calls (default 1 = serial)

    Returns:
        Enriched DataFrame
    """
    # Use a sample if specified
    if sample_size is not None:
        df_sample = (
            df.sample(sample_size, random_state=42) if sample_size < len(df) else df
        )
    else:
        df_sample = df

    # Get texts to analyze
    texts = df_sample["full_text"].tolist()

    # Get LLM analysis
    responses = get_llm_analysis(texts, model=model, api_key=api_key, workers=workers)

    # Extract content from responses
    contents = [extract_llm_content(response) for response in responses]

    # Create a copy of the DataFrame to avoid modifying the original
    df_enriched = df_sample.copy()

    # Add LLM analysis to the DataFrame
    df_enriched["llm_analysis"] = contents

    # Save enriched dataset if output path is provided
    if output_path:
        df_enriched.to_parquet(output_path, index=False)
        print(f"Enriched dataset saved to {output_path}")

    return df_enriched
