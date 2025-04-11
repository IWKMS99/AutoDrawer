import logging
import numpy as np
import mss
import cv2
from typing import Optional, Tuple, Dict


class ScreenCapturer:
    """
    Класс для захвата экрана или его областей с использованием mss.
    Возвращает изображения в формате RGB.
    """

    def __init__(self):
        """Инициализация захвата экрана."""
        self.logger = logging.getLogger(__name__)
        self.logger.debug("ScreenCapturer инициализирован.")

    def capture_fullscreen(self) -> Optional[np.ndarray]:
        """
        Захватывает весь экран.

        Returns:
            Optional[np.ndarray]: Изображение в формате RGB или None при ошибке.
        """
        try:
            with mss.mss() as sct:
                # Захватываем монитор 0 (весь экран, включая все мониторы)
                monitor = sct.monitors[0]
                sct_img = sct.grab(monitor)
            # Преобразуем в numpy массив, отбрасываем альфа-канал и конвертируем в RGB
            img = np.array(sct_img)[:, :, :3]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.logger.debug(f"Захвачен полный экран: {img_rgb.shape}")
            return img_rgb
        except Exception as e:
            self.logger.error(f"Ошибка захвата полного экрана: {e}")
            return None

    def capture_area(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Захватывает указанную область экрана.

        Args:
            top_left: Координаты верхнего левого угла (x, y).
            bottom_right: Координаты нижнего правого угла (x, y).

        Returns:
            Optional[np.ndarray]: Изображение области в формате RGB или None при ошибке.
        """
        try:
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]
            if width <= 0 or height <= 0:
                self.logger.error(f"Неверные размеры области: {width}x{height}")
                return None

            monitor: Dict[str, int] = {
                "top": top_left[1],
                "left": top_left[0],
                "width": width,
                "height": height
            }
            self.logger.debug(f"Захват области: {monitor}")

            with mss.mss() as sct:
                sct_img = sct.grab(monitor)
            img = np.array(sct_img)[:, :, :3]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.logger.debug(f"Захвачена область: {img_rgb.shape}")
            return img_rgb
        except Exception as e:
            self.logger.error(f"Ошибка захвата области {top_left} -> {bottom_right}: {e}")
            return None
