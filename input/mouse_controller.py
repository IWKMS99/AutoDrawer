import logging
import time
import pyautogui
from typing import Optional, Tuple
from pynput import mouse

from core.state_manager import StateManager
from utils.config_loader import ConfigLoader


class MouseController:
    """
    Класс для управления мышью: клики, выбор областей, выбор цвета.
    """

    def __init__(self, state: StateManager, config: ConfigLoader):
        """
        Инициализация контроллера мыши.

        Args:
            state: Экземпляр StateManager для проверки состояния.
            config: Экземпляр ConfigLoader для доступа к настройкам.
        """
        self.logger = logging.getLogger(__name__)
        self.state = state
        self.config = config
        self.click_listener: Optional[mouse.Listener] = None
        self.click_coords: Optional[Tuple[int, int]] = None
        self.area_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

        # Настройка pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = config.get("failsafe_delay", 0.01)
        self.logger.debug("MouseController инициализирован.")

    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> bool:
        """
        Обработчик кликов мыши для слушателя.

        Args:
            x: Координата X.
            y: Координата Y.
            button: Кнопка мыши.
            pressed: Нажата (True) или отпущена (False).

        Returns:
            bool: False для остановки слушателя, True для продолжения.
        """
        if not pressed or button != mouse.Button.left or not self.state.is_running():
            return True
        self.click_coords = (x, y)
        self.logger.debug(f"Зафиксирован клик: {self.click_coords}")
        return False  # Останавливаем слушатель после клика

    def get_click(self, prompt: str) -> Optional[Tuple[int, int]]:
        """
        Ожидает клик мыши и возвращает его координаты.

        Args:
            prompt: Сообщение для пользователя.

        Returns:
            Optional[Tuple[int, int]]: Координаты клика или None, если отменено.
        """
        self.click_coords = None
        print(f"\n--> {prompt}")
        self.logger.info(f"Ожидание клика: {prompt}")

        with mouse.Listener(on_click=self._on_click) as listener:
            self.click_listener = listener
            listener.join()

        if not self.state.is_running():
            self.logger.warning("Получение клика отменено (программа остановлена).")
            return None

        coords = self.click_coords
        self.click_listener = None
        if coords:
            self.logger.info(f"Получены координаты клика: {coords}")
        else:
            self.logger.warning("Клик не зафиксирован.")
        return coords

    def get_area(self, prompt: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Запрашивает выбор области двумя кликами.

        Args:
            prompt: Сообщение для пользователя.

        Returns:
            Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
                Координаты верхнего левого и нижнего правого углов или (None, None).
        """
        self.area_coords = None
        print(f"\n--> {prompt}")
        self.logger.info(f"Ожидание выбора области: {prompt}")

        # Первый клик (верхний левый угол)
        top_left = self.get_click("Кликните верхний левый угол области.")
        if not top_left or not self.state.is_running():
            self.logger.warning("Выбор области отменен на первом клике.")
            return None, None

        # Второй клик (нижний правый угол)
        bottom_right = self.get_click("Кликните нижний правый угол области.")
        if not bottom_right or not self.state.is_running():
            self.logger.warning("Выбор области отменен на втором клике.")
            return None, None

        # Проверка корректности координат
        if top_left[0] >= bottom_right[0] or top_left[1] >= bottom_right[1]:
            self.logger.error(f"Некорректная область: TL={top_left}, BR={bottom_right}")
            return None, None

        self.logger.info(f"Область выбрана: TL={top_left}, BR={bottom_right}")
        return top_left, bottom_right

    def click(self, x: int, y: int, delay: Optional[float] = None) -> bool:
        """
        Выполняет клик мыши в указанной точке.

        Args:
            x: Координата X.
            y: Координата Y.
            delay: Задержка после клика (сек), если None — используется failsafe_delay.

        Returns:
            bool: True, если клик успешен, False при ошибке или остановке.
        """
        if not self.state.is_running():
            self.logger.warning(f"Клик ({x}, {y}) отменен: программа остановлена.")
            return False

        if self.state.is_paused():
            self.logger.debug(f"Ожидание возобновления перед кликом ({x}, {y})...")
            if not self.state.wait_while_paused():
                return False

        try:
            self.logger.debug(f"Клик мыши: ({x}, {y})")
            pyautogui.click(x, y)
            time.sleep(delay if delay is not None else self.config.get("failsafe_delay", 0.01))
            return True
        except Exception as e:
            self.logger.exception(f"Ошибка клика ({x}, {y}): {e}")
            return False

    def select_color(self, position: Tuple[int, int]) -> bool:
        """
        Выбирает цвет на палитре кликом.

        Args:
            position: Координаты цвета на палитре.

        Returns:
            bool: True, если выбор успешен, False при ошибке или остановке.
        """
        self.logger.debug(f"Выбор цвета на позиции {position}...")
        if not self.click(position[0], position[1], delay=self.config.get("color_change_delay", 0.08)):
            self.logger.error(f"Не удалось выбрать цвет на {position}.")
            return False
        return True
