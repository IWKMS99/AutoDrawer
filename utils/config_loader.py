import logging
import yaml
from typing import Any, Dict


class ConfigLoader:
    """
    Класс для загрузки, валидации и управления конфигурацией.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Инициализация загрузчика конфигурации.

        Args:
            config_path: Путь к файлу конфигурации (по умолчанию config/settings.yaml).
        """
        self.logger = logging.getLogger(__name__)
        self.config: Dict[str, Any] = self._load_config(config_path)
        self._validate_config()
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
            "failsafe_delay": 0.01,
            "canvas_detection": {
                "search_offset_y_ratio": 0.3,
                "adaptive_thresh_block_size": 11,
                "adaptive_thresh_c": 2,
                "morph_kernel_size": 3,
                "morph_iterations": 1,
                "cell_min_aspect_ratio": 0.8,
                "cell_max_aspect_ratio": 1.2,
                "cell_min_width": 5,
                "cell_max_width": 100,
                "cell_min_height": 5,
                "cell_max_height": 100,
                "assume_portrait": True,
                "fallback_cell_size": 20,
            }
        }
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f) or {}
            self._update_nested_dict(default_config, loaded_config)
            self.logger.info(f"Конфигурация загружена из {path}")
        except FileNotFoundError:
            self.logger.warning(f"Файл конфигурации {path} не найден. Используются значения по умолчанию.")
        except Exception as e:
            self.logger.exception(f"Ошибка загрузки конфигурации из {path}: {e}")
        return default_config

    def _update_nested_dict(self, default: Dict, update: Dict) -> None:
        """
        Рекурсивно обновляет словарь default значениями из update.

        Args:
            default: Исходный словарь.
            update: Словарь с новыми значениями.
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                self._update_nested_dict(default[key], value)
            else:
                default[key] = value

    def _validate_config(self) -> None:
        """
        Проверяет корректность конфигурации, логирует предупреждения при некорректных значениях.
        """
        # Проверка основных параметров
        if not isinstance(self.config.get("target_width"), int) or self.config["target_width"] <= 0:
            self.logger.warning("Некорректное значение target_width. Установлено по умолчанию: 100")
            self.config["target_width"] = 100
        if not isinstance(self.config.get("num_colors"), int) or self.config["num_colors"] <= 0:
            self.logger.warning("Некорректное значение num_colors. Установлено по умолчанию: 10")
            self.config["num_colors"] = 10
        for delay in ["click_delay", "color_change_delay", "clear_click_delay", "failsafe_delay"]:
            if not isinstance(self.config.get(delay), (int, float)) or self.config[delay] <= 0:
                self.logger.warning(f"Некорректное значение {delay}. Установлено по умолчанию: 0.5")
                self.config[delay] = 0.5

        # Проверка параметров canvas_detection
        canvas = self.config.get("canvas_detection", {})
        validations = {
            "search_offset_y_ratio": (float, lambda x: 0.0 <= x <= 1.0, 0.3),
            "adaptive_thresh_block_size": (int, lambda x: x > 3 and x % 2 == 1, 11),
            "adaptive_thresh_c": (int, lambda x: 0 <= x <= 10, 2),
            "morph_kernel_size": (int, lambda x: x > 0 and x % 2 == 1, 3),
            "morph_iterations": (int, lambda x: x >= 1, 1),
            "cell_min_aspect_ratio": (float, lambda x: 0.5 <= x <= 1.0, 0.8),
            "cell_max_aspect_ratio": (float, lambda x: 1.0 <= x <= 2.0, 1.2),
            "cell_min_width": (int, lambda x: x > 0, 5),
            "cell_max_width": (int, lambda x: x > canvas.get("cell_min_width", 5), 100),
            "cell_min_height": (int, lambda x: x > 0, 5),
            "cell_max_height": (int, lambda x: x > canvas.get("cell_min_height", 5), 100),
            "assume_portrait": (bool, lambda x: True, True),
            "fallback_cell_size": (int, lambda x: x > 0, 20),
        }
        for key, (type_, condition, default) in validations.items():
            value = canvas.get(key)
            if not isinstance(value, type_) or not condition(value):
                self.logger.warning(f"Некорректное значение canvas_detection.{key}. Установлено: {default}")
                canvas[key] = default

    def get(self, key: str, default: Any = None) -> Any:
        """
        Возвращает значение из конфигурации.

        Args:
            key: Ключ настройки (поддерживает вложенные ключи через точку, например, 'canvas_detection.search_offset_y_ratio').
            default: Значение по умолчанию, если ключ не найден.

        Returns:
            Any: Значение настройки или default.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, {})
            if value == {}:
                self.logger.debug(f"Ключ {key} не найден. Возвращено: {default}")
                return default
        if value == {}:
            value = default
        self.logger.debug(f"Получено значение: {key} = {value}")
        return value

    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Обновляет конфигурацию новыми значениями.

        Args:
            new_config: Словарь с новыми значениями.
        """
        self._update_nested_dict(self.config, new_config)
        self._validate_config()
        self.logger.debug(f"Конфигурация обновлена: {new_config}")
