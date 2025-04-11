import logging
from typing import Optional


class InputValidator:
    """
    Класс для валидации пользовательского ввода (числа, задержки).
    """

    def __init__(self):
        """Инициализация валидатора ввода."""
        self.logger = logging.getLogger(__name__)
        self.logger.debug("InputValidator инициализирован.")

    def get_positive_int(self, prompt: str, default: Optional[int] = None) -> int:
        """
        Запрашивает положительное целое число.

        Args:
            prompt: Сообщение для пользователя.
            default: Значение по умолчанию (если None, ввод обязателен).

        Returns:
            int: Введенное число или значение по умолчанию.
        """
        while True:
            try:
                value = input(
                    f"{prompt} (Enter={'по умолчанию ' + str(default) if default is not None else 'обязательно'}): ").strip()
                if not value and default is not None:
                    self.logger.debug(f"Использовано значение по умолчанию: {default}")
                    return default
                num = int(value)
                if num <= 0:
                    print("Ошибка: введите положительное число.")
                    self.logger.warning(f"Введено неположительное число: {num}")
                    continue
                self.logger.debug(f"Введено корректное число: {num}")
                return num
            except ValueError:
                print("Ошибка: введите целое число.")
                self.logger.warning(f"Некорректный ввод: {value}")

    def get_float(self, prompt: str, default: Optional[float] = None, non_negative: bool = False) -> float:
        """
        Запрашивает число с плавающей точкой.

        Args:
            prompt: Сообщение для пользователя.
            default: Значение по умолчанию (если None, ввод обязателен).
            non_negative: Требовать неотрицательное значение.

        Returns:
            float: Введенное число или значение по умолчанию.
        """
        while True:
            try:
                value = input(
                    f"{prompt} (Enter={'по умолчанию ' + str(default) if default is not None else 'обязательно'}): ").strip()
                if not value and default is not None:
                    self.logger.debug(f"Использовано значение по умолчанию: {default}")
                    return default
                num = float(value)
                if non_negative and num < 0:
                    print("Ошибка: введите неотрицательное число.")
                    self.logger.warning(f"Введено отрицательное число: {num}")
                    continue
                self.logger.debug(f"Введено корректное число: {num}")
                return num
            except ValueError:
                print("Ошибка: введите число.")
                self.logger.warning(f"Некорректный ввод: {value}")
