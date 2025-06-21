# FakeNews-X-PT-BR Dataset

## Descrição

O **FakeNews-X-PT-BR** é um conjunto de dados criado para pesquisa acadêmica em detecção de notícias falsas em domínios cruzados, especificamente para o idioma português brasileiro. Este dataset contém tweets coletados da plataforma X (anteriormente Twitter) classificados como verdadeiros ou falsos, abrangendo diferentes categorias temáticas.

## Características do Dataset

- **Total de amostras**: 660 tweets
- **Distribuição balanceada**: 330 notícias falsas e 330 verdadeiras
- **Idioma**: Português brasileiro
- **Período de coleta**: 2018-2024
- **Fonte**: Plataforma X (Twitter)

### Categorias Temáticas

| Categoria | Quantidade |
|-----------|------------|
| Política | 435 |
| Saúde | 143 |
| Outros | 44 |
| Ambiente | 22 |
| Economia | 8 |
| Celebridades | 8 |

## Estrutura dos Dados

O dataset está disponível no arquivo `dataset.parquet` com as seguintes colunas:

### Informações do Tweet
- `tweet_id`: ID único do tweet
- `full_text`: Texto completo do tweet
- `created_at`: Data e hora de criação do tweet
- `categoria`: Categoria temática do conteúdo
- `label`: Classificação da notícia ("fake" ou "true")

### Métricas de Engajamento
- `favorite_count`: Número de curtidas do tweet
- `retweet_count`: Número de retweets do tweet
- `reply_count`: Número de respostas ao tweet

### Informações do Perfil
- `description`: Biografia do usuário
- `verified`: Status de verificação da conta (True/False)
- `followers_count`: Número de seguidores
- `friends_count`: Número de pessoas seguidas
- `statuses_count`: Número de tweets publicados pelo usuário
- `favourites_count`: Número de tweets curtidos pelo usuário

## Aplicações

Este dataset pode ser utilizado para:

- Desenvolvimento de modelos de detecção de notícias falsas
- Pesquisas em processamento de linguagem natural
- Análise de comportamento em redes sociais
- Estudos sobre desinformação no contexto brasileiro
- Desenvolvimento de sistemas de fact-checking
- Pesquisa em domínios cruzados (cross-domain)

## Considerações Éticas

- Este dataset foi coletado para fins exclusivamente acadêmicos
- Os dados respeitam as políticas de uso da plataforma X
- Recomenda-se o uso responsável para combater a desinformação
- Não deve ser utilizado para criar ou propagar notícias falsas

## Como Citar

Se você utilizar este dataset em sua pesquisa, por favor cite adequadamente o trabalho e forneça a devida atribuição conforme a licença CC BY 4.0.

## Licença

Este conjunto de dados está licenciado sob a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Contato

Para questões sobre o dataset ou colaborações em pesquisa, entre em contato através do e-mail: rafael.braz@usp.br.

---

**Nota**: Este dataset foi criado para fins de pesquisa acadêmica. O uso dos dados deve seguir as diretrizes éticas de pesquisa e as políticas de uso da plataforma de origem.
