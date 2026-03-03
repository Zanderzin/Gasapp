"""Script principal do pipeline ETL."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.etl.downloader import download_anp_data
from src.etl.transformer import run_transform
from src.etl.loader import load_to_db, load_to_parquet
from loguru import logger
from datetime import datetime

def main():
    logger.info("=== Iniciando pipeline ETL ===")
    raw_file = download_anp_data()
    df = run_transform(raw_file)
    load_to_db(df)
    ts = datetime.now().strftime("%Y%m%d")
    load_to_parquet(df, f"data/processed/precos_semanais_{ts}.parquet")
    logger.success("=== Pipeline ETL concluido ===")

if __name__ == "__main__":
    main()
