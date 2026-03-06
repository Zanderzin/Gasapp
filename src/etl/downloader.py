"""
Download automatico dos dados semanais de precos de combustiveis da ANP.

Logica:
- Gera as datas de todas as semanas (domingo a sabado) dentro do intervalo desejado
- Tenta baixar cada arquivo pelo padrao de URL da ANP
- Controla arquivos ja baixados via arquivo de registro (downloaded.txt)
- Salva tudo em data/raw/
"""
import requests
import time
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta


BASE_URL = "https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/arquivos-lpc"
DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
REGISTRY_FILE = DATA_RAW_DIR / "downloaded.txt"


def get_week_ranges(start_date: datetime, end_date: datetime) -> list:
    """
    Gera lista de tuplas (inicio_semana, fim_semana) entre duas datas.
    A ANP usa semanas de domingo a sabado.
    """
    weeks = []
    current = start_date

    # Ajusta para o domingo mais proximo anterior
    current -= timedelta(days=current.weekday() + 1)
    if current < start_date:
        current += timedelta(weeks=1)

    while current <= end_date:
        week_start = current
        week_end = current + timedelta(days=6)
        weeks.append((week_start, week_end))
        current += timedelta(weeks=1)

    return weeks


def build_url(week_start: datetime, week_end: datetime) -> str:
    """Monta a URL do arquivo Excel da ANP para uma semana especifica."""
    year = week_start.year
    start_str = week_start.strftime("%Y-%m-%d")
    end_str = week_end.strftime("%Y-%m-%d")
    filename = f"revendas_lpc_{start_str}_{end_str}.xlsx"
    return f"{BASE_URL}/{year}/{filename}", filename


def load_registry() -> set:
    """Carrega o conjunto de arquivos ja baixados."""
    if not REGISTRY_FILE.exists():
        return set()
    return set(REGISTRY_FILE.read_text(encoding="utf-8").splitlines())


def save_registry(registry: set) -> None:
    """Salva o registro atualizado de arquivos baixados."""
    REGISTRY_FILE.write_text("\n".join(sorted(registry)), encoding="utf-8")


def download_file(url: str, filename: str, save_dir: Path) -> bool:
    """
    Baixa um arquivo da ANP e salva localmente.
    Retorna True se baixou com sucesso, False se nao encontrou.
    """
    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 404:
            return False

        response.raise_for_status()

        filepath = save_dir / filename
        filepath.write_bytes(response.content)
        logger.success(f"Baixado: {filename} ({len(response.content) / 1024:.1f} KB)")
        return True

    except requests.HTTPError as e:
        logger.warning(f"Erro HTTP {e.response.status_code}: {filename}")
        return False
    except requests.RequestException as e:
        logger.error(f"Erro de conexao ao baixar {filename}: {e}")
        return False


def download_range(
    years_back: int = 3,
    end_date: datetime = None,
    save_dir: Path = DATA_RAW_DIR,
    delay_seconds: float = 1.0,
) -> dict:
    """
    Baixa todos os arquivos semanais da ANP dentro do intervalo definido.

    Args:
        years_back: Quantos anos para tras baixar. Padrao: 3 anos.
        end_date: Data final do intervalo. Padrao: hoje.
        save_dir: Diretorio onde salvar os arquivos.
        delay_seconds: Pausa entre requisicoes para nao sobrecarregar o servidor.

    Returns:
        Dicionario com contagens: baixados, ignorados, erros.
    """
    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=365 * years_back)
    save_dir.mkdir(parents=True, exist_ok=True)

    weeks = get_week_ranges(start_date, end_date)
    registry = load_registry()

    logger.info(f"Intervalo: {start_date.date()} ate {end_date.date()}")
    logger.info(f"Total de semanas a verificar: {len(weeks)}")
    logger.info(f"Arquivos ja baixados: {len(registry)}")

    stats = {"baixados": 0, "ignorados": 0, "nao_encontrados": 0, "erros": 0}

    for week_start, week_end in weeks:
        url, filename = build_url(week_start, week_end)

        # Pula se ja foi baixado
        if filename in registry:
            logger.debug(f"Ignorado (ja existe): {filename}")
            stats["ignorados"] += 1
            continue

        success = download_file(url, filename, save_dir)

        if success:
            registry.add(filename)
            save_registry(registry)
            stats["baixados"] += 1
            time.sleep(delay_seconds)
        else:
            stats["nao_encontrados"] += 1

    logger.info(f"Resultado: {stats}")
    return stats


def download_latest(save_dir: Path = DATA_RAW_DIR) -> bool:
    """
    Baixa apenas o arquivo da semana mais recente.
    Util para o cron job semanal apos a carga historica inicial.
    """
    today = datetime.now()
    # Semana atual: volta para o domingo anterior
    days_since_sunday = (today.weekday() + 1) % 7
    week_start = today - timedelta(days=days_since_sunday)
    week_end = week_start + timedelta(days=6)

    save_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry()

    url, filename = build_url(week_start, week_end)

    if filename in registry:
        logger.info(f"Semana atual ja baixada: {filename}")
        return False

    success = download_file(url, filename, save_dir)
    if success:
        registry.add(filename)
        save_registry(registry)

    return success


if __name__ == "__main__":
    # Carga historica inicial: baixa os ultimos 3 anos
    download_range(years_back=3)
