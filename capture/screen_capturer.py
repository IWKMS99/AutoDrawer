import logging
import numpy as np
import mss
import cv2
from typing import Optional, Tuple, Dict, List


class ScreenCapturer:
    """
    Класс для захвата экрана или его областей с использованием mss.
    Возвращает изображения в формате RGB.
    """

    def __init__(self):
        """Инициализация захвата экрана."""
        self.logger = logging.getLogger(__name__)
        self.sct = mss.mss()
        self.monitors = self._get_monitors()
        self.selected_monitor = None  # Номер монитора (по умолчанию None)
        self.logger.debug("ScreenCapturer инициализирован.")

    def _get_monitors(self) -> List[Dict[str, int]]:
        """Возвращает список доступных мониторов."""
        monitors = self.sct.monitors
        self.logger.debug(f"Обнаружено мониторов: {len(monitors)}")
        return monitors

    def set_monitor(self, monitor_idx: int) -> bool:
        """
        Устанавливает монитор для захвата.

        Args:
            monitor_idx: Индекс монитора (0 - все экраны, 1+ - конкретные мониторы).

        Returns:
            bool: True если монитор выбран успешно, False если индекс некорректен.
        """
        if 0 <= monitor_idx < len(self.monitors):
            self.selected_monitor = monitor_idx
            monitor = self.monitors[monitor_idx]
            self.logger.info(f"Выбран монитор {monitor_idx}: {monitor}")
            return True
        self.logger.error(f"Некорректный индекс монитора: {monitor_idx}. Доступно: {len(self.monitors)}")
        return False

    def capture_fullscreen(self) -> Optional[np.ndarray]:
        """
        Захватывает весь экран выбранного монитора.

        Returns:
            Optional[np.ndarray]: Изображение в формате RGB или None при ошибке.
        """
        try:
            monitor_idx = self.selected_monitor if self.selected_monitor is not None else 0
            monitor = self.monitors[monitor_idx]
            sct_img = self.sct.grab(monitor)
            # Преобразуем в numpy массив, отбрасываем альфа-канал и конвертируем в RGB
            img = np.array(sct_img)[:, :, :3]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.logger.debug(f"Захвачен экран монитора {monitor_idx}: {img_rgb.shape}")
            return img_rgb
        except Exception as e:
            self.logger.error(f"Ошибка захвата экрана монитора {monitor_idx}: {e}")
            return None

    def capture_area(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Захватывает указанную область экрана на выбранном мониторе.

        Args:
            top_left: Координаты верхнего левого угла (x, y).
            bottom_right: Координаты нижнего правого угла (x, y).

        Returns:
            Optional[np.ndarray]: Изображение области в формате RGB или None при ошибке.
        """
        try:
            monitor_idx = self.selected_monitor if self.selected_monitor is not None else 0
            monitor = self.monitors[monitor_idx]
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]
            if width <= 0 or height <= 0:
                self.logger.error(f"Неверные размеры области: {width}x{height}")
                return None

            # Корректируем координаты относительно выбранного монитора
            adjusted_top_left = (top_left[0] - monitor["left"], top_left[1] - monitor["top"])
            monitor_area = {
                "top": monitor["top"] + adjusted_top_left[1],
                "left": monitor["left"] + adjusted_top_left[0],
                "width": width,
                "height": height
            }
            self.logger.debug(f"Захват области на мониторе {monitor_idx}: {monitor_area}")

            sct_img = self.sct.grab(monitor_area)
            img = np.array(sct_img)[:, :, :3]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.logger.debug(f"Захвачена область: {img_rgb.shape}")
            return img_rgb
        except Exception as e:
            self.logger.error(f"Ошибка захвата области {top_left} -> {bottom_right} на мониторе {monitor_idx}: {e}")
            return None
