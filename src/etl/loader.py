"""
Salva os dados processados em Parquet para uso nos notebooks e modelos.
"""
import pandas as pd
from pathlib import Path
from loguru import logger


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def save_parquet(df: pd.DataFrame, filename: str = "precos_df.parquet") -> Path:
    """
    Salva o DataFrame processado em Parquet.

    Args:
        df: DataFrame limpo e transformado.
        filename: Nome do arquivo de saida.

    Returns:
        Path do arquivo salvo.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DIR / filename

    df.to_parquet(filepath, index=False)

    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.success(f"Parquet salvo: {filepath.name} | {len(df)} registros | {size_mb:.2f} MB")
    return filepath


def load_parquet(filename: str = "precos_df.parquet") -> pd.DataFrame:
    """
    Carrega o Parquet processado.
    Util para usar nos notebooks e modelos sem reprocessar tudo.

    Args:
        filename: Nome do arquivo a carregar.

    Returns:
        DataFrame carregado.
    """
    filepath = PROCESSED_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")

    df = pd.read_parquet(filepath)
    logger.info(f"Parquet carregado: {filepath.name} | {len(df)} registros")
    return df


if __name__ == "__main__":
    from transformer import run_transform_all

    df = run_transform_all()
    save_parquet(df)
