import logging
from typing import Optional
from pynput import keyboard

from core.state_manager import StateManager


class KeyboardListener:
    """
    Класс для прослушивания клавиатуры и управления состоянием программы.
    """

    def __init__(self, state: StateManager):
        """
        Инициализация слушателя клавиатуры.

        Args:
            state: Экземпляр StateManager для управления состоянием.
        """
        self.logger = logging.getLogger(__name__)
        self.state = state
        self.listener: Optional[keyboard.Listener] = None
        self.logger.debug("KeyboardListener инициализирован.")

    def _on_press(self, key: keyboard.Key) -> Optional[bool]:
        """
        Обработчик нажатий клавиш.

        Args:
            key: Нажатая клавиша.

        Returns:
            Optional[bool]: False для остановки слушателя, None/True для продолжения.
        """
        try:
            if key == keyboard.Key.esc:
                self.state.stop()
                self.logger.info("Остановка по нажатию ESC.")
                return False
            elif key == keyboard.Key.space:
                self.state.toggle_pause()
                return True
        except Exception as e:
            self.logger.exception(f"Ошибка обработки клавиши {key}: {e}")
            return True

    def start(self) -> None:
        """Запускает слушатель клавиатуры."""
        if self.listener is None or not self.listener.running:
            self.listener = keyboard.Listener(on_press=self._on_press)
            self.listener.start()
            self.logger.info("Слушатель клавиатуры запущен.")

    def stop(self) -> None:
        """Останавливает слушатель клавиатуры."""
        if self.listener and self.listener.running:
            self.listener.stop()
            self.listener = None
            self.logger.info("Слушатель клавиатуры остановлен.")
