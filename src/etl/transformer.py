"""
Limpeza e transformacao dos dados brutos da ANP.

Estrutura do arquivo Excel da ANP (revendas_lpc):
- Linhas 1-9: cabecalho institucional (ignorar)
- Linha 10: nomes das colunas
- Linha 11 em diante: dados por posto de combustivel
"""
import pandas as pd
from pathlib import Path
from loguru import logger


COLUMNS_MAP = {
    "CNPJ": "cnpj",
    "RAZÃO": "razao_social",
    "FANTASIA": "nome_fantasia",
    "ENDEREÇO": "endereco",
    "NÚMERO": "numero",
    "COMPLEMENTO": "complemento",
    "BAIRRO": "bairro",
    "CEP": "cep",
    "MUNICÍPIO": "municipio",
    "ESTADO": "estado",
    "BANDEIRA": "bandeira",
    "PRODUTO": "produto",
    "UNIDADE DE MEDIDA": "unidade",
    "PREÇO DE REVENDA": "preco_revenda",
    "DATA DA COLETA": "data_coleta",
}


def load_raw(filepath: Path) -> pd.DataFrame:
    """
    Carrega um arquivo Excel da ANP.
    Pula as 9 primeiras linhas de cabecalho institucional.
    """
    df = pd.read_excel(filepath, skiprows=9, dtype=str)
    logger.info(f"Carregado: {Path(filepath).name} | {len(df)} linhas")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia colunas, converte tipos e remove linhas invalidas.
    """
    # Renomeia colunas
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMNS_MAP)

    # Mantem apenas colunas conhecidas
    df = df[[c for c in COLUMNS_MAP.values() if c in df.columns]]

    # Remove linhas completamente vazias
    df = df.dropna(how="all")

    # Converte data
    df["data_coleta"] = pd.to_datetime(df["data_coleta"], dayfirst=True, errors="coerce")

    # Converte preco (virgula -> ponto)
    df["preco_revenda"] = pd.to_numeric(
        df["preco_revenda"].astype(str).str.replace(",", ".").str.strip(),
        errors="coerce"
    )

 # Remove datas futuras (erros de digitacao ANP)
    hoje = pd.Timestamp.now().normalize()
    n_futuro = (df["data_coleta"] > hoje).sum()
    if n_futuro > 0:
        logger.warning(f"Removendo {n_futuro} registros com data futura")
        df = df[df["data_coleta"] <= hoje]

    # Marca GLP (vendido em botijao 13kg, nao em litro) para nao misturar com combustiveis liquidos
    df["is_botijao"] = df["produto"] == "GLP"
    
    # Substitui NAN no nome fantasia pela razao social
    df["nome_fantasia"] = df.apply(
        lambda r: r["razao_social"] if str(r["nome_fantasia"]).upper() in ["NAN", "", "NONE"] else r["nome_fantasia"],
        axis=1
    )

    # Padroniza texto
    text_cols = ["estado", "municipio", "produto", "bandeira", "unidade", "razao_social", "nome_fantasia", "bairro"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # Filtra apenas Distrito Federal
    df = df[df["estado"] == "DISTRITO FEDERAL"]

    # Extrai regiao a partir do campo bairro
    # Ex: "AREA DE DESENVOLVIMENTO ECONOMICO (AGUAS CLARAS)" -> regiao = "AGUAS CLARAS"
    # Ex: "AGUAS CLARAS" -> regiao = "AGUAS CLARAS"
    def extrair_regiao(bairro):
        import re
        match = re.search(r'\((.+?)\)', str(bairro))
        if match:
            return match.group(1).strip()
        return str(bairro).strip()

    df['regiao'] = df['bairro'].apply(extrair_regiao)

    # Normaliza grafias inconsistentes de regioes
    normalizacao = {
        # Aguas Claras
        "AGUAS CLARAS":         "AGUAS CLARAS",
        "AGUAAS CLARAS":        "AGUAS CLARAS",
        "AGUAS CLARAS":         "AGUAS CLARAS",
        # Ceilandia
        "CEILANDIA":            "CEILANDIA",
        "CEILANDIA NORTE":      "CEILANDIA NORTE",
        # Gama
        "GAMA-DF":              "GAMA",
        # Samambaia
        "SAMABAIA":             "SAMAMBAIA",
        "SAMAMBIA":             "SAMAMBAIA",
        "SAMAMBAIA - SUL":      "SAMAMBAIA SUL",
        # Taguatinga
        "TAGUATINGA.":          "TAGUATINGA",
        # Nucleo Bandeirante
        "N BANDEIRANTE":        "NUCLEO BANDEIRANTE",
        # Areal Aguas Claras (sem parenteses, regiao errada)
        "AREAL AGUAS CLARAS":   "AGUAS CLARAS",
    }
    df['regiao'] = df['regiao'].str.strip()

    # Remove acentos para padronizar
    import unicodedata
    def remove_acentos(texto):
        return unicodedata.normalize('NFKD', str(texto)).encode('ascii', 'ignore').decode('ascii')

    df['regiao'] = df['regiao'].apply(remove_acentos).str.upper().str.strip()
    df['regiao'] = df['regiao'].replace(normalizacao)

    # Padroniza CNPJ e CEP (so numeros)
    for col in ["cnpj", "cep"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r"\D", "", regex=True)

    logger.info(f"Apos limpeza: {len(df)} linhas validas")
    return df


def run_transform(filepath: Path) -> pd.DataFrame:
    """Executa o pipeline completo de transformacao para um arquivo."""
    df = load_raw(filepath)
    df = clean(df)
    return df


def run_transform_all(raw_dir: Path = None) -> pd.DataFrame:
    """
    Carrega e transforma todos os arquivos Excel do diretorio,
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

    # Remove duplicatas (mesmo posto + produto + data)
    combined = combined.drop_duplicates(subset=["cnpj", "produto", "data_coleta"])
    combined = combined.sort_values(["produto", "estado", "municipio", "data_coleta"])
    combined = combined.reset_index(drop=True)

    logger.success(
        f"Total combinado: {len(combined)} registros | "
        f"{combined['produto'].nunique()} produtos | "
        f"{combined['estado'].nunique()} estados | "
        f"{combined['cnpj'].nunique()} postos unicos"
    )
    return combined


if __name__ == "__main__":
    df = run_transform_all()
    print(df.head(10))
    print(f"\nColunas: {df.columns.tolist()}")
    print(f"\nProdutos: {df['produto'].unique()}")
    print(f"\nPeriodo: {df['data_coleta'].min()} ate {df['data_coleta'].max()}")






