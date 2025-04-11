import logging
import numpy as np
import mss
import cv2
from typing import Optional, Tuple, Dict, List


class ScreenCapturer:
    """Класс для захвата экрана с поддержкой множественных мониторов"""

    DEFAULT_MONITOR_INDEX = 0  # Основной монитор по умолчанию

    def __init__(self):
        """Инициализация захватчика экрана"""
        self.logger = logging.getLogger(__name__)
        self.sct = mss.mss()
        self.monitors = self._detect_monitors()
        self.selected_monitor_idx: Optional[int] = None
        self.logger.debug("ScreenCapturer инициализирован")

    def _detect_monitors(self) -> List[Dict[str, int]]:
        """
        Обнаруживает доступные мониторы

        Returns:
            Список словарей с информацией о мониторах
        """
        monitors = self.sct.monitors
        if not monitors:
            self.logger.warning("Мониторы не обнаружены")
        else:
            self.logger.debug(f"Обнаружено мониторов: {len(monitors)}")
        return monitors

    def set_monitor(self, monitor_idx: int) -> bool:
        """
        Выбирает монитор для захвата

        Args:
            monitor_idx: Индекс монитора (начиная с 0)

        Returns:
            True при успешном выборе, иначе False
        """
        if self._is_valid_monitor_index(monitor_idx):
            self.selected_monitor_idx = monitor_idx
            monitor_info = self.monitors[monitor_idx]
            self.logger.info(f"Выбран монитор {monitor_idx}: "
                             f"{monitor_info['width']}x{monitor_info['height']} "
                             f"({monitor_info['left']},{monitor_info['top']})")
            return True
        return False

    def _is_valid_monitor_index(self, index: int) -> bool:
        """
        Проверяет корректность индекса монитора

        Args:
            index: Проверяемый индекс

        Returns:
            True если индекс валиден, иначе False
        """
        is_valid = 0 <= index < len(self.monitors)
        if not is_valid:
            self.logger.error(f"Некорректный индекс монитора: {index}. "
                              f"Доступно: {len(self.monitors)}")
        return is_valid

    def capture_fullscreen(self) -> Optional[np.ndarray]:
        """
        Захватывает весь экран выбранного монитора

        Returns:
            RGB-изображение в формате numpy array, или None при ошибке
        """
        self.logger.debug("Выполняется захват всего экрана")
        return self._capture_region()

    def capture_area(self,
                     top_left: Tuple[int, int],
                     bottom_right: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Захватывает указанную область на выбранном мониторе

        Args:
            top_left: Координаты левого верхнего угла (x, y)
            bottom_right: Координаты правого нижнего угла (x, y)

        Returns:
            RGB-изображение в формате numpy array, или None при ошибке

        Пример:
            capturer = ScreenCapturer()
            capturer.set_monitor(0)
            image = capturer.capture_area((100, 100), (500, 500))
        """
        self.logger.debug(f"Захват области: {top_left} -> {bottom_right}")

        monitor = self._get_current_monitor()
        if not monitor:
            return None

        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]

        if not self._validate_dimensions(width, height):
            return None

        region = self._calculate_region_coordinates(top_left, width, height, monitor)
        if not region:
            return None

        return self._capture_region(**region)

    def _capture_region(self, **kwargs) -> Optional[np.ndarray]:
        """
        Базовый метод захвата области с обработкой ошибок

        Args:
            **kwargs: Параметры области захвата (top, left, width, height)

        Returns:
            RGB-изображение или None при ошибке
        """
        try:
            monitor = self._get_current_monitor()
            if not monitor:
                return None

            # Используем текущий монитор если регион не указан
            region = kwargs if kwargs else monitor
            self.logger.debug(f"Захват региона: {region}")

            screenshot = self.sct.grab(region)
            return self._convert_image(screenshot)

        except Exception as e:
            self.logger.error(f"Ошибка захвата: {str(e)} | Регион: {kwargs}")
            return None

    def _get_current_monitor(self) -> Optional[Dict[str, int]]:
        """
        Возвращает текущий выбранный монитор или основной

        Returns:
            Информация о мониторе или None
        """
        if self.selected_monitor_idx is not None:
            return self.monitors[self.selected_monitor_idx]
        return self.monitors[self.DEFAULT_MONITOR_INDEX] if self.monitors else None

    def _validate_dimensions(self, width: int, height: int) -> bool:
        """
        Проверяет корректность размеров области

        Args:
            width: Ширина области
            height: Высота области

        Returns:
            True если размеры валидны, иначе False
        """
        if width <= 0 or height <= 0:
            self.logger.error(f"Неверные размеры области: {width}x{height}")
            return False
        return True

    def _calculate_region_coordinates(self,
                                      top_left: Tuple[int, int],
                                      width: int,
                                      height: int,
                                      monitor: Dict[str, int]) -> Optional[Dict[str, int]]:
        """
        Вычисляет абсолютные координаты области с проверкой границ

        Args:
            top_left: Координаты левого верхнего угла относительно экрана
            width: Ширина области
            height: Высота области
            monitor: Информация о текущем мониторе

        Returns:
            Словарь с абсолютными координатами или None при ошибке
        """
        try:
            # Корректируем координаты относительно монитора
            left = top_left[0] - monitor['left']
            top = top_left[1] - monitor['top']

            # Проверяем выход за границы
            if any([
                left < 0,
                top < 0,
                left + width > monitor['width'],
                top + height > monitor['height']
            ]):
                raise ValueError("Область выходит за пределы монитора")

            return {
                'top': monitor['top'] + top,
                'left': monitor['left'] + left,
                'width': width,
                'height': height
            }

        except Exception as e:
            self.logger.error(f"Ошибка расчета координат: {str(e)}")
            return None

    def _convert_image(self, screenshot: mss.screenshot.ScreenShot) -> np.ndarray:
        """
        Конвертирует изображение из формата mss в RGB-массив

        Args:
            screenshot: Объект скриншота от mss

        Returns:
            RGB-массив в формате numpy
        """
        # Удаляем альфа-канал и конвертируем цвета
        image = np.array(screenshot)[:, :, :3]
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
