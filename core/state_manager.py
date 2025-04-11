import threading
import logging
import time
from typing import Optional


class StateManager:
    """
    Управляет состоянием программы: запуск, пауза, остановка.
    Обеспечивает потокобезопасный доступ к флагам состояния.
    """

    def __init__(self):
        """Инициализация состояния программы."""
        self.logger = logging.getLogger(__name__)
        self._running: bool = True
        self._paused: bool = False
        self._lock = threading.Lock()
        self.logger.debug("StateManager инициализирован.")

    def is_running(self) -> bool:
        """
        Проверяет, запущена ли программа.

        Returns:
            bool: True, если программа работает, иначе False.
        """
        with self._lock:
            return self._running

    def is_paused(self) -> bool:
        """
        Проверяет, приостановлена ли программа.

        Returns:
            bool: True, если программа на паузе, иначе False.
        """
        with self._lock:
            return self._paused

    def stop(self) -> None:
        """Останавливает программу."""
        with self._lock:
            if self._running:
                self._running = False
                self._paused = False
                self.logger.info("Программа остановлена.")

    def toggle_pause(self) -> None:
        """Переключает состояние паузы."""
        with self._lock:
            self._paused = not self._paused
            status = "приостановлена" if self._paused else "возобновлена"
            self.logger.info(f"Программа {status}.")

    def wait_while_paused(self) -> bool:
        """
        Ожидает, пока программа не выйдет из паузы или не остановится.

        Returns:
            bool: True, если программа возобновлена, False, если остановлена.
        """
        while self.is_paused():
            if not self.is_running():
                self.logger.warning("Ожидание прервано: программа остановлена.")
                return False
            time.sleep(0.1)
        return self.is_running()
