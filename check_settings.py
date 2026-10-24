import os

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)  # создаёт папку logs, если её нет
import os
import sys

# Добавляем корень проекта в начало поиска модулей
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.core.config import get_settings

s = get_settings()
print("PROJECT_NAME:", s.PROJECT_NAME)
print("ollama_url:", s.ollama_url)
print("ASYNC_DATABASE_URL:", s.ASYNC_DATABASE_URL)
print("ensemble_models_list:", s.ensemble_models_list)
