# src/logger.py
import logging
import os

# Cria o diretório de logs se não existir
os.makedirs("logs", exist_ok=True)

# Configura o logger
logging.basicConfig(
    filename="logs/app.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger()
