"""
Script principal do pipeline ETL.
- Uso inicial: baixa historico + processa + salva
- Uso semanal (cron): baixa nova semana + atualiza Parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.etl.downloader import download_range, download_latest
from src.etl.transformer import run_transform_all
from src.etl.loader import save_parquet
from loguru import logger
import argparse


def run(mode: str = "update"):
    """
    Executa o pipeline ETL completo.

    Args:
        mode: 
            'full'   — carga historica completa (primeira vez)
            'update' — baixa so a semana mais recente (uso semanal)
    """
    logger.info(f"=== Iniciando pipeline ETL | modo={mode} ===")

    # 1. Download
    if mode == "full":
        logger.info("Modo full: baixando historico dos ultimos 3 anos...")
        download_range(years_back=3)
    else:
        logger.info("Modo update: baixando semana mais recente...")
        download_latest()

    # 2. Transformacao
    logger.info("Transformando arquivos...")
    df = run_transform_all()

    if df.empty:
        logger.error("Nenhum dado processado. Abortando.")
        return

    # 3. Carga
    save_parquet(df)

    logger.success("=== Pipeline ETL concluido ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ETL - Fuel Predictor")
    parser.add_argument(
        "--mode",
        choices=["full", "update"],
        default="update",
        help="'full' para carga historica, 'update' para semana atual (padrao)"
    )
    args = parser.parse_args()
    run(mode=args.mode)
