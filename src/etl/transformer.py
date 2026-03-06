"""
Limpeza e transformacao dos dados brutos da ANP.

Estrutura do arquivo Excel da ANP:
- Linhas 1-9: cabecalho institucional (ignorar)
- Linha 10: nomes das colunas
- Linha 11 em diante: dados
"""
import pandas as pd
from pathlib import Path
from loguru import logger


COLUMNS_MAP = {
    "DATA INICIAL": "data_inicial",
    "DATA FINAL": "data_final",
    "ESTADO": "estado",
    "MUNICÍPIO": "municipio",
    "PRODUTO": "produto",
    "NÚMERO DE POSTOS PESQUISADOS": "n_postos",
    "UNIDADE DE MEDIDA": "unidade",
    "PREÇO MÉDIO REVENDA": "preco_medio",
    "DESVIO PADRÃO REVENDA": "desvio_padrao",
    "PREÇO MÍNIMO REVENDA": "preco_min",
    "PREÇO MÁXIMO REVENDA": "preco_max",
    "COEF DE VARIAÇÃO REVENDA": "coef_variacao",
}


def load_raw(filepath: Path) -> pd.DataFrame:
    """
    Carrega um arquivo Excel da ANP.
    Pula as 9 primeiras linhas de cabecalho institucional.
    A linha 10 vira o cabecalho das colunas.
    """
    df = pd.read_excel(filepath, skiprows=9, dtype=str)
    logger.info(f"Carregado: {Path(filepath).name} | {len(df)} linhas")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia colunas, converte tipos e remove linhas invalidas.
    """
    # Renomeia colunas (strip para remover espacos extras)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMNS_MAP)

    # Mantem apenas colunas conhecidas
    df = df[[c for c in COLUMNS_MAP.values() if c in df.columns]]

    # Remove linhas completamente vazias
    df = df.dropna(how="all")

    # Converte datas
    df["data_inicial"] = pd.to_datetime(df["data_inicial"], dayfirst=True, errors="coerce")
    df["data_final"] = pd.to_datetime(df["data_final"], dayfirst=True, errors="coerce")

    # Converte colunas numericas (virgula -> ponto)
    numeric_cols = ["preco_medio", "desvio_padrao", "preco_min", "preco_max", "coef_variacao", "n_postos"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".").str.strip(),
                errors="coerce"
            )

    # Remove linhas sem data ou sem preco medio
    df = df.dropna(subset=["data_inicial", "preco_medio"])
    df = df[df["preco_medio"] > 0]

    # Padroniza texto
    for col in ["estado", "municipio", "produto", "unidade"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    logger.info(f"Apos limpeza: {len(df)} linhas validas")
    return df


def run_transform(filepath: Path) -> pd.DataFrame:
    """Executa o pipeline completo de transformacao para um arquivo."""
    df = load_raw(filepath)
    df = clean(df)
    return df


def run_transform_all(raw_dir: Path = None) -> pd.DataFrame:
    """
    Carrega e transforma todos os arquivos Excel de um diretorio,
    concatenando tudo em um unico DataFrame.
    """
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"

    files = sorted(raw_dir.glob("*.xlsx"))
    logger.info(f"Arquivos encontrados: {len(files)}")

    dfs = []
    for filepath in files:
        try:
            df = run_transform(filepath)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Erro ao processar {filepath.name}: {e}")

    if not dfs:
        logger.error("Nenhum arquivo processado com sucesso.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicatas (mesma semana + municipio + produto)
    combined = combined.drop_duplicates(subset=["data_inicial", "estado", "municipio", "produto"])
    combined = combined.sort_values(["produto", "estado", "municipio", "data_inicial"])
    combined = combined.reset_index(drop=True)

    logger.success(f"Total combinado: {len(combined)} registros | {combined['produto'].nunique()} produtos | {combined['estado'].nunique()} estados")
    return combined


if __name__ == "__main__":
    df = run_transform_all()
    print(df.head(10))
    print(f"\nColunas: {df.columns.tolist()}")
    print(f"\nProdutos: {df['produto'].unique()}")
    print(f"\nPeriodo: {df['data_inicial'].min()} ate {df['data_inicial'].max()}")
