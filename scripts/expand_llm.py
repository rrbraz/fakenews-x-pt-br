import json
from pathlib import Path

import pandas as pd


def expand_llm_analysis(input_parquet_path, output_parquet_path):
    """
    Carrega um arquivo Parquet, expande uma coluna JSON em múltiplas colunas
    e salva o resultado em um novo arquivo Parquet.
    """
    try:
        df = pd.read_parquet(input_parquet_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo de entrada não encontrado em {input_parquet_path}")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo Parquet: {e}")
        return

    if "llm_analysis" not in df.columns:
        print("Erro: Coluna 'llm_analysis' não encontrada no arquivo de entrada.")
        return

    # Função para carregar JSON de forma segura, tratando erros e NaNs
    def safe_json_loads(json_str):
        if pd.isna(json_str):
            return {}
        try:
            # Tentativa de correção para JSONs com aspas simples ou problemas de escape
            if isinstance(json_str, str):
                return json.loads(json_str)
            return {}
        except json.JSONDecodeError:
            return {}

    parsed_json_column = df["llm_analysis"].apply(safe_json_loads)

    # Verifica se a coluna parseada está vazia (todos os JSONs foram malformados ou NaN)
    if all(not d for d in parsed_json_column):
        print(
            "Aviso: A coluna 'llm_analysis' resultou em dados vazios após o parsing. Verifique o formato do JSON."
        )
        # Salva o dataframe original se não houver dados JSON para processar, ou um com colunas vazias.
        # Neste caso, vamos prosseguir para que colunas vazias 'llm_*' sejam criadas se json_normalize retornar df vazio.

    normalized_data = pd.json_normalize(parsed_json_column)

    # Adiciona prefixo às colunas normalizadas, apenas se normalized_data não estiver vazio
    if not normalized_data.empty:
        normalized_data = normalized_data.add_prefix("llm_")
    else:
        # Se normalized_data estiver vazio (nenhum JSON válido), cria um DataFrame vazio com prefixo para evitar erros
        # ou simplesmente não concatena nada se for preferível.
        # Para garantir que o concat não falhe, podemos criar um df vazio com o mesmo índice.
        normalized_data = pd.DataFrame(index=df.index)

    # Combina com o DataFrame original, removendo a coluna JSON original
    df_expanded = pd.concat(
        [df.drop(columns=["llm_analysis"]), normalized_data], axis=1
    )

    # Pós-processamento para colunas que são listas

    # Para 'llm_entidades_nomeadas'
    entidades_col_name = "llm_entidades_nomeadas"
    if entidades_col_name in df_expanded.columns:
        df_expanded["llm_entidades_nomes"] = df_expanded[entidades_col_name].apply(
            lambda x: (
                ", ".join(
                    sorted(
                        list(
                            set(
                                e.get("nome", "")
                                for e in x
                                if isinstance(e, dict) and e.get("nome")
                            )
                        )
                    )
                )
                if isinstance(x, list) and x
                else None
            )
        )
        df_expanded["llm_entidades_tipos"] = df_expanded[entidades_col_name].apply(
            lambda x: (
                ", ".join(
                    sorted(
                        list(
                            set(
                                e.get("tipo", "")
                                for e in x
                                if isinstance(e, dict) and e.get("tipo")
                            )
                        )
                    )
                )
                if isinstance(x, list) and x
                else None
            )
        )
        df_expanded = df_expanded.drop(columns=[entidades_col_name])

    # Para 'llm_reivindicacoes_centrais'
    reivindicacoes_col_name = "llm_reivindicacoes_centrais"
    if reivindicacoes_col_name in df_expanded.columns:
        df_expanded["llm_reivindicacoes_texto"] = df_expanded[
            reivindicacoes_col_name
        ].apply(
            lambda x: (
                " | ".join(
                    item[0]
                    for item in x
                    if isinstance(item, list)
                    and len(item) > 0
                    and isinstance(item[0], str)
                )
                if isinstance(x, list) and x
                else None
            )
        )
        df_expanded["llm_reivindicacoes_status"] = df_expanded[
            reivindicacoes_col_name
        ].apply(
            lambda x: (
                ", ".join(
                    item[1]
                    for item in x
                    if isinstance(item, list)
                    and len(item) > 1
                    and isinstance(item[1], str)
                )
                if isinstance(x, list) and x
                else None
            )
        )
        df_expanded = df_expanded.drop(columns=[reivindicacoes_col_name])

    # Para listas de strings como 'llm_tom', 'llm_marcadores_linguisticos', 'llm_fonte_mencionada'
    list_columns_to_join = [
        "llm_tom",
        "llm_marcadores_linguisticos",
        "llm_fonte_mencionada",
    ]
    for col_name in list_columns_to_join:
        if col_name in df_expanded.columns:
            # Garante que apenas listas sejam processadas com join, e strings/NaNs sejam mantidos
            df_expanded[col_name] = df_expanded[col_name].apply(
                lambda x: (
                    ", ".join(sorted(list(set(str(i) for i in x))))
                    if isinstance(x, list) and x
                    else (str(x) if pd.notna(x) and not isinstance(x, list) else None)
                )
            )

    # A coluna 'llm_sinais_de_incerteza' pode ser uma string ou uma lista.
    sinais_incerteza_col_name = "llm_sinais_de_incerteza"
    if sinais_incerteza_col_name in df_expanded.columns:
        df_expanded[sinais_incerteza_col_name] = df_expanded[
            sinais_incerteza_col_name
        ].apply(
            lambda x: (
                ", ".join(sorted(list(set(str(i) for i in x))))
                if isinstance(x, list) and x
                else (str(x) if pd.notna(x) and not isinstance(x, list) else None)
            )
        )

    try:
        Path(output_parquet_path).parent.mkdir(parents=True, exist_ok=True)
        df_expanded.to_parquet(output_parquet_path, index=False)
        print(f"Arquivo expandido salvo com sucesso em: {output_parquet_path}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo Parquet: {e}")


if __name__ == "__main__":
    # Caminhos dos arquivos de entrada e saída
    input_parquet = "data/generated/dataset_enriched.parquet"
    output_parquet = "data/generated/dataset_expanded.parquet"

    print(f"Iniciando expansão do arquivo: {input_parquet}")
    expand_llm_analysis(input_parquet, output_parquet)
    print("Processo de expansão concluído.")
