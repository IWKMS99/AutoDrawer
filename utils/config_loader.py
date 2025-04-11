import logging
import yaml
from typing import Any, Dict


class ConfigLoader:
    """
    Класс для загрузки и управления конфигурацией.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Инициализация загрузчика конфигурации.

        Args:
            config_path: Путь к файлу конфигурации (по умолчанию config/settings.yaml).
        """
        self.logger = logging.getLogger(__name__)
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.logger.debug("ConfigLoader инициализирован.")

    def _load_config(self, path: str) -> Dict[str, Any]:
        """
        Загружает конфигурацию из YAML-файла или возвращает значения по умолчанию.

        Args:
            path: Путь к файлу конфигурации.

        Returns:
            Dict[str, Any]: Словарь с настройками.
        """
        default_config = {
            "target_width": 100,
            "num_colors": 10,
            "click_delay": 0.5,
            "color_change_delay": 0.08,
            "clear_click_delay": 0.5,
            "failsafe_delay": 0.01
        }
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f) or {}
            default_config.update(loaded_config)
            self.logger.info(f"Конфигурация загружена из {path}: {default_config}")
        except FileNotFoundError:
            self.logger.warning(f"Файл конфигурации {path} не найден. Используются значения по умолчанию.")
        except Exception as e:
            self.logger.exception(f"Ошибка загрузки конфигурации из {path}: {e}")
        return default_config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Возвращает значение из конфигурации.

        Args:
            key: Ключ настройки.
            default: Значение по умолчанию, если ключ не найден.

        Returns:
            Any: Значение настройки или default.
        """
        value = self.config.get(key, default)
        self.logger.debug(f"Получено значение: {key} = {value}")
        return value

    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Обновляет конфигурацию новыми значениями.

        Args:
            new_config: Словарь с новыми значениями.
        """
        self.config.update(new_config)
        self.logger.debug(f"Конфигурация обновлена: {new_config}")
