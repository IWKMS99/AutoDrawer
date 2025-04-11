import logging
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple, List, Set

from core.state_manager import StateManager
from utils.debug_tools import DebugTools
from image_processing.kmeans_handler import KMeansHandler


class ImageProcessor:
    """
    Класс для обработки входного изображения: уменьшение, предпросмотр, извлечение пикселей.
    """

    def __init__(self, debug: bool = False):
        """
        Инициализация процессора изображений.

        Args:
            debug: Включение режима отладки с OpenCV.
        """
        self.logger = logging.getLogger(__name__)
        self.debug_tools = DebugTools(debug)
        self.kmeans_handler = KMeansHandler()
        self.logger.debug("ImageProcessor инициализирован.")

    def downsample_image(self, path: str, target_width: int, target_height: int) -> Tuple[
        Optional[np.ndarray], int, int]:
        """
        Уменьшает изображение до целевого размера с сохранением пропорций.

        Args:
            path: Путь к файлу изображения.
            target_width: Целевая ширина (в пикселях или ячейках).
            target_height: Целевая высота (в пикселях или ячейках).

        Returns:
            Tuple[Optional[np.ndarray], int, int]: Уменьшенное изображение, его ширина и высота, или (None, 0, 0) при ошибке.
        """
        self.logger.info(f"Уменьшение изображения {path} до {target_width}x{target_height}...")
        try:
            with Image.open(path) as img:
                # Преобразование в RGB (удаляем альфа-канал, если есть)
                img_rgb = img.convert("RGB")
                img_thumb = img_rgb.copy()

                # Уменьшение с сохранением пропорций
                img_thumb.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
                img_array = np.array(img_thumb)

                # Коррекция ориентации, если ширина больше высоты
                if img_array.shape[1] > img_array.shape[0]:
                    img_array = np.rot90(img_array)
                    self.logger.info("Изображение повернуто на 90° для соответствия холсту.")

                self.logger.info(f"Итоговый размер: {img_array.shape[1]}x{img_array.shape[0]}")
                if self.debug_tools.enabled:
                    self.debug_tools.show_image("Downsampled Image", img_array)

                return img_array, img_array.shape[1], img_array.shape[0]
        except Exception as e:
            self.logger.exception(f"Ошибка уменьшения изображения {path}: {e}")
            return None, 0, 0

    def preview_image(self, path: str, target_width: int, target_height: int, state: StateManager) -> bool:
        """
        Создает и показывает предпросмотр изображения, запрашивает подтверждение.

        Args:
            path: Путь к файлу изображения.
            target_width: Целевая ширина.
            target_height: Целевая высота.
            state: Экземпляр StateManager для проверки состояния.

        Returns:
            bool: True, если пользователь подтвердил, False, если отменил или произошла ошибка.
        """
        self.logger.info("Создание предпросмотра изображения...")
        print("\n--- Предпросмотр ---")
        try:
            # Уменьшение изображения
            downsampled_img, w, h = self.downsample_image(path, target_width, target_height)
            if downsampled_img is None:
                self.logger.error("Не удалось создать предпросмотр: изображение не уменьшено.")
                return False

            # Кластеризация цветов (для точного предпросмотра)
            pixels = self.get_input_pixels(path, target_width, target_height, num_colors=10)[0]
            if not pixels:
                self.logger.error("Не удалось извлечь пиксели для предпросмотра.")
                return False

            # Создание изображения предпросмотра
            preview_img = np.zeros((h, w, 3), dtype=np.uint8)
            for x, y, color in pixels:
                preview_img[y, x] = color

            self.debug_tools.show_image("Предпросмотр", preview_img, wait_key=False)
            self.logger.info(f"Показан предпросмотр: {w}x{h}")

            print("Проверьте предпросмотр:")
            print("- Нажмите ПРОБЕЛ для продолжения.")
            print("- Нажмите ESC для отмены.")
            while state.is_running():
                key = cv2.waitKey(100)
                if key == 32:  # Пробел
                    self.logger.info("Предпросмотр подтвержден пользователем.")
                    cv2.destroyWindow("Предпросмотр")
                    return True
                elif key == 27:  # ESC
                    self.logger.info("Предпросмотр отклонен пользователем.")
                    cv2.destroyWindow("Предпросмотр")
                    return False
                elif state.is_paused():
                    self.logger.info("Программа на паузе во время предпросмотра.")
                    print("(На паузе. Нажмите Пробел для продолжения...)")
                    state.wait_while_paused()

            self.logger.warning("Предпросмотр прерван (программа остановлена).")
            cv2.destroyWindow("Предпросмотр")
            return False

        except Exception as e:
            self.logger.exception(f"Ошибка предпросмотра: {e}")
            return False

    def get_input_pixels(self, path: str, target_width: int, target_height: int, num_colors: int) -> Tuple[
        List[Tuple[int, int, Tuple[int, int, int]]], int, int]:
        """
        Извлекает пиксели изображения с кластеризацией цветов.

        Args:
            path: Путь к файлу изображения.
            target_width: Целевая ширина.
            target_height: Целевая высота.
            num_colors: Количество цветов для кластеризации.

        Returns:
            Tuple[List[Tuple[int, int, Tuple[int, int, int]]], int, int]:
                Список (x, y, цвет RGB), ширина и высота изображения, или ([], 0, 0) при ошибке.
        """
        self.logger.info(f"Извлечение пикселей из {path} с {num_colors} цветами...")
        try:
            # Уменьшение изображения
            img, w, h = self.downsample_image(path, target_width, target_height)
            if img is None:
                return [], 0, 0

            # Подготовка пикселей для кластеризации
            pixels = img.reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)
            num_clusters = min(num_colors, len(unique_colors))
            self.logger.info(f"Уникальных цветов: {len(unique_colors)}, кластеров: {num_clusters}")

            # Кластеризация
            cluster_centers, labels = self.kmeans_handler.cluster_colors(pixels, num_clusters)
            cluster_centers = cluster_centers.astype(int)

            # Формирование списка пикселей
            input_pixels = []
            for i in range(h):
                for j in range(w):
                    pixel_idx = i * w + j
                    color = tuple(cluster_centers[labels[pixel_idx]])
                    input_pixels.append((j, i, color))

            self.logger.info(f"Извлечено {len(input_pixels)} пикселей, цветов: {num_clusters}")
            return input_pixels, w, h

        except Exception as e:
            self.logger.exception(f"Ошибка извлечения пикселей: {e}")
            return [], 0, 0
