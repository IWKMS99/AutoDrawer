import logging
import numpy as np
import cv2
from typing import Optional, List, Tuple

from capture.screen_capturer import ScreenCapturer
from utils.debug_tools import DebugTools
from image_processing.kmeans_handler import KMeansHandler


class PaletteAnalyzer:
    """
    Класс для анализа палитры: определение области, извлечение цветов и позиции ластика.
    """

    def __init__(self, capturer: ScreenCapturer, debug: bool = False):
        """
        Инициализация анализатора палитры.

        Args:
            capturer: Экземпляр ScreenCapturer для захвата экрана.
            debug: Включение режима отладки с OpenCV.
        """
        self.logger = logging.getLogger(__name__)
        self.capturer = capturer
        self.debug_tools = DebugTools(debug)
        self.kmeans_handler = KMeansHandler()
        self.palette_top_left: Optional[Tuple[int, int]] = None
        self.palette_bottom_right: Optional[Tuple[int, int]] = None
        self.circles: List[Tuple[int, int, int]] = []  # (x, y, radius)
        self.eraser_pos: Optional[Tuple[int, int]] = None
        self.logger.debug("PaletteAnalyzer инициализирован.")

    def detect_palette(self) -> Tuple[
        Optional[Tuple[int, int]], Optional[Tuple[int, int]], List[Tuple[int, int, int]], Optional[Tuple[int, int]]
    ]:
        """
        Автоматическое определение области палитры и кругов.

        Returns:
            Tuple: (top_left, bottom_right, circles, eraser_pos) или (None, None, [], None) при ошибке.
        """
        self.logger.info("Попытка автоопределения палитры...")
        try:
            full_img = self.capturer.capture_fullscreen()
            if full_img is None:
                self.logger.error("Не удалось захватить экран для определения палитры.")
                return None, None, [], None

            h, w, _ = full_img.shape
            # Ограничиваем область поиска верхними 10% и левой третью экрана
            search_region = full_img[0:int(h * 0.1), 0:int(w * 0.33)]
            self.debug_tools.show_image("Palette Detect - Search Region", search_region)

            # Преобразование и размытие для поиска кругов
            gray = cv2.cvtColor(search_region, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            self.debug_tools.show_image("Palette Detect - Gray Blurred", gray)

            # Поиск кругов с помощью HoughCircles
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                param1=50, param2=20, minRadius=23, maxRadius=25
            )

            if circles is None:
                self.logger.error("Круги палитры не найдены.")
                self.debug_tools.show_image("Palette Detect - No Circles", gray)
                return None, None, [], None

            circles_uint = np.uint16(np.around(circles[0]))
            self.logger.info(f"Найдено {len(circles_uint)} кругов до фильтрации.")

            # Ограничение до 10 кругов, сортировка по X
            circles_sorted = sorted(circles_uint, key=lambda c: c[0])[:10]
            self.logger.info(f"Ограничено до {len(circles_sorted)} кругов.")

            # Преобразование в абсолютные координаты
            final_circles_data: List[Tuple[int, int, int]] = []
            circles_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            for i in circles_sorted:
                rel_x, rel_y, radius = i[0], i[1], i[2]
                abs_x = rel_x  # Относительно полного экрана (координаты уже в системе экрана)
                abs_y = rel_y
                final_circles_data.append((abs_x, abs_y, radius))
                cv2.circle(circles_img, (rel_x, rel_y), radius, (0, 255, 0), 2)
                cv2.circle(circles_img, (rel_x, rel_y), 1, (0, 0, 255), 3)

            self.debug_tools.show_image("Palette Detect - Found Circles", circles_img)

            # Определение ластика (самый правый круг)
            eraser_center = (final_circles_data[-1][0], final_circles_data[-1][1]) if final_circles_data else None

            # Определение границ палитры
            all_x = [c[0] - c[2] for c in final_circles_data] + [c[0] + c[2] for c in final_circles_data]
            all_y = [c[1] - c[2] for c in final_circles_data] + [c[1] + c[2] for c in final_circles_data]
            palette_top_left = (min(all_x), min(all_y))
            palette_bottom_right = (max(all_x), max(all_y))

            if self.debug_tools.enabled:
                debug_final_img = full_img.copy()
                rel_tl = (palette_top_left[0], palette_top_left[1])
                rel_br = (palette_bottom_right[0], palette_bottom_right[1])
                cv2.rectangle(debug_final_img, rel_tl, rel_br, (0, 255, 0), 3)
                for rx, ry, r in final_circles_data:
                    cv2.circle(debug_final_img, (rx, ry), r, (0, 255, 0), 1)
                if eraser_center:
                    cv2.circle(debug_final_img, eraser_center, 5, (0, 0, 255), -1)
                self.debug_tools.show_image("Palette Detect - Final Area", debug_final_img)

            self.logger.info(
                f"Палитра: TL={palette_top_left}, BR={palette_bottom_right}, Кругов: {len(final_circles_data)}")
            self.logger.info(f"Предполагаемый ластик: {eraser_center}")
            return palette_top_left, palette_bottom_right, final_circles_data, eraser_center

        except Exception as e:
            self.logger.exception(f"Ошибка автоопределения палитры: {e}")
            return None, None, [], None

    def set_palette(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int],
                    circles: List[Tuple[int, int, int]]) -> None:
        """
        Устанавливает координаты палитры и круги вручную.

        Args:
            top_left: Координаты верхнего левого угла.
            bottom_right: Координаты нижнего правого угла.
            circles: Список кругов (x, y, radius).
        """
        self.palette_top_left = top_left
        self.palette_bottom_right = bottom_right
        self.circles = circles
        self.logger.info(f"Палитра установлена вручную: TL={top_left}, BR={bottom_right}, Кругов: {len(circles)}")

    def set_eraser(self, eraser_pos: Optional[Tuple[int, int]]) -> None:
        """
        Устанавливает позицию ластика.

        Args:
            eraser_pos: Координаты ластика или None.
        """
        self.eraser_pos = eraser_pos
        if eraser_pos:
            self.logger.info(f"Ластик установлен: {eraser_pos}")
        else:
            self.logger.debug("Ластик не установлен (eraser_pos=None).")

    def capture_palette(self) -> Optional[np.ndarray]:
        """
        Захватывает изображение области палитры.

        Returns:
            Optional[np.ndarray]: Изображение палитры в RGB или None при ошибке.
        """
        if not self.palette_top_left or not self.palette_bottom_right:
            self.logger.error("Область палитры не задана.")
            return None
        return self.capturer.capture_area(self.palette_top_left, self.palette_bottom_right)

    def extract_colors(self, palette_img: np.ndarray, num_colors_expected: int) -> List[
        Tuple[Tuple[int, int, int], Tuple[int, int]]]:
        """
        Извлекает цвета и их позиции из палитры.

        Args:
            palette_img: Изображение палитры в RGB.
            num_colors_expected: Ожидаемое количество цветов.

        Returns:
            List[Tuple[Tuple[int, int, int], Tuple[int, int]]]: Список (цвет RGB, позиция клика).
        """
        if palette_img is None or self.palette_top_left is None:
            self.logger.error("Изображение палитры или координаты не заданы.")
            return []

        if self.circles:
            self.logger.info(f"Извлечение цветов на основе {len(self.circles)} кругов...")
            palette_colors_data = []
            debug_palette_img = palette_img.copy() if self.debug_tools.enabled else None

            for i, (circle_x, circle_y, radius) in enumerate(self.circles):
                rel_x = circle_x - self.palette_top_left[0]
                rel_y = circle_y - self.palette_top_left[1]
                sample_radius = max(1, radius // 4)
                y_start = max(0, rel_y - sample_radius)
                y_end = min(palette_img.shape[0], rel_y + sample_radius + 1)
                x_start = max(0, rel_x - sample_radius)
                x_end = min(palette_img.shape[1], rel_x + sample_radius + 1)

                if y_start >= y_end or x_start >= x_end:
                    self.logger.warning(f"Круг {i + 1} дал пустую область: ({x_start}: {x_end}, {y_start}: {y_end})")
                    continue

                sample_area = palette_img[y_start:y_end, x_start:x_end]
                if sample_area.size == 0:
                    self.logger.warning(f"Круг {i + 1} дал пустую область (size=0).")
                    continue

                avg_color = tuple(np.mean(sample_area, axis=(0, 1)).astype(int))
                click_pos = (circle_x, circle_y)
                palette_colors_data.append((avg_color, click_pos))
                self.logger.debug(f"Круг {i + 1}: Центр={click_pos}, Цвет={avg_color}")

                if debug_palette_img is not None:
                    cv2.rectangle(debug_palette_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
                    cv2.circle(debug_palette_img, (rel_x, rel_y), 3, (0, 0, 255), -1)

            if debug_palette_img is not None:
                self.debug_tools.show_image("Palette Extract - From Circles", debug_palette_img)

            if len(palette_colors_data) != num_colors_expected:
                self.logger.warning(
                    f"Найдено {len(palette_colors_data)} цветов, ожидалось {num_colors_expected}."
                )
            return palette_colors_data

        else:
            self.logger.warning("Круги не найдены. Используем K-Means.")
            return self._extract_colors_kmeans(palette_img, num_colors_expected)

    def _extract_colors_kmeans(self, palette_img: np.ndarray, num_colors_expected: int, sample_points: int = 500) -> \
            List[Tuple[Tuple[int, int, int], Tuple[int, int]]]:
        """
        Извлечение цветов с помощью K-Means (fallback).

        Args:
            palette_img: Изображение палитры в RGB.
            num_colors_expected: Ожидаемое количество цветов.
            sample_points: Количество точек для сэмплирования.

        Returns:
            List[Tuple[Tuple[int, int, int], Tuple[int, int]]]: Список (цвет RGB, позиция клика).
        """
        self.logger.info(f"Извлечение цветов K-Means (ожидается {num_colors_expected})...")
        try:
            height, width, _ = palette_img.shape
            pixels_list = palette_img.reshape(-1, 3)
            if pixels_list.shape[0] > sample_points:
                indices = np.random.choice(pixels_list.shape[0], sample_points, replace=False)
                sampled_colors = pixels_list[indices, :]
            else:
                sampled_colors = pixels_list

            unique_colors = np.unique(sampled_colors, axis=0)
            num_clusters = min(num_colors_expected, len(unique_colors)) if len(
                unique_colors) < num_colors_expected else num_colors_expected

            cluster_centers, labels = self.kmeans_handler.cluster_colors(sampled_colors, num_clusters)
            cluster_centers = cluster_centers.astype(int)

            palette_coords = np.array([(y, x) for y in range(height) for x in range(width)])
            all_labels = self.kmeans_handler.kmeans_handler.predict(pixels_list)

            palette_colors_data = []
            processed_centers = set()
            debug_palette_img = palette_img.copy() if self.debug_tools.enabled else None

            for i in range(num_clusters):
                center_color_tuple = tuple(cluster_centers[i])
                if center_color_tuple in processed_centers:
                    continue

                cluster_pixel_indices = np.where(all_labels == i)[0]
                if len(cluster_pixel_indices) == 0:
                    self.logger.debug(f"Кластер {i} пуст.")
                    continue

                cluster_coords = palette_coords[cluster_pixel_indices]
                avg_y, avg_x = np.mean(cluster_coords, axis=0).astype(int)
                screen_pos = (self.palette_top_left[0] + avg_x, self.palette_top_left[1] + avg_y)

                palette_colors_data.append((center_color_tuple, screen_pos))
                processed_centers.add(center_color_tuple)
                self.logger.debug(f"KMeans Кластер {i}: Цвет={center_color_tuple}, Позиция={screen_pos}")

                if debug_palette_img is not None:
                    cv2.circle(debug_palette_img, (avg_x, avg_y), 5, (0, 0, 255), -1)

            if debug_palette_img is not None:
                self.debug_tools.show_image("Palette Extract - KMeans Fallback", debug_palette_img)

            return palette_colors_data

        except Exception as e:
            self.logger.exception(f"Ошибка K-Means: {e}")
            return []
