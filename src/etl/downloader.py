"""Download automatico dos dados da ANP."""
import requests
from pathlib import Path
from loguru import logger
from datetime import datetime

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def download_anp_data(year=None, save_dir=DATA_RAW_DIR):
    if year is None:
        year = datetime.now().year
    save_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Ajustar URL conforme estrutura atual do portal ANP
    url = f"https://www.gov.br/anp/pt-br/assuntos/precos/arquivos/semana-{year}.xlsx"
    logger.info(f"Iniciando download ANP: ano={year}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    filename = save_dir / f"anp_{year}_{datetime.now().strftime('%Y%m%d')}.xlsx"
    filename.write_bytes(response.content)
    logger.success(f"Download concluido: {filename}")
    return filename

if __name__ == "__main__":
    download_anp_data()
