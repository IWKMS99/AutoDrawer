import logging
import cv2
import numpy as np
from typing import Optional


class DebugTools:
    """
    Класс для инструментов отладки с OpenCV.
    """

    def __init__(self, debug: bool = False):
        """
        Инициализация инструментов отладки.

        Args:
            debug: Включает отладочные функции, если True.
        """
        self.logger = logging.getLogger(__name__)
        self.enabled = debug
        self.windows_open = set()
        if debug:
            self.logger.debug("DebugTools включены.")
        else:
            self.logger.debug("DebugTools отключены.")

    def show_image(self, window_name: str, image: np.ndarray, wait_key: bool = True) -> None:
        """
        Показывает изображение в окне OpenCV.

        Args:
            window_name: Название окна.
            image: Изображение (RGB или grayscale).
            wait_key: Ожидать нажатия клавиши (True) или показать без блокировки (False).
        """
        if not self.enabled:
            return

        if image is None:
            self.logger.debug(f"[{window_name}] Изображение отсутствует.")
            return

        try:
            # Преобразование в BGR для OpenCV, если RGB
            display_img = image.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow(window_name, display_img)
            self.windows_open.add(window_name)
            self.logger.debug(f"Показ изображения: {window_name}, размер={image.shape}")

            if wait_key:
                key = cv2.waitKey(0)
                if key == ord('q') or key == 27:  # 'q' или ESC
                    self.cleanup()
                else:
                    cv2.destroyWindow(window_name)
                    self.windows_open.discard(window_name)
        except Exception as e:
            self.logger.exception(f"Ошибка отображения {window_name}: {e}")

    def cleanup(self) -> None:
        """Закрывает все открытые окна OpenCV."""
        if not self.enabled:
            return

        for window in list(self.windows_open):
            cv2.destroyWindow(window)
        cv2.destroyAllWindows()
        self.windows_open.clear()
        self.logger.debug("Все окна OpenCV закрыты.")
