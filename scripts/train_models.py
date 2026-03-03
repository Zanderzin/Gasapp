"""Script de treinamento dos modelos."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loguru import logger

def main():
    logger.info("=== Treinamento ===")
    logger.warning("Implementacao disponivel na Fase 2.")

if __name__ == "__main__":
    main()
