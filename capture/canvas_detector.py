import logging
import numpy as np
import cv2
from typing import Optional, Tuple, List

from capture.screen_capturer import ScreenCapturer
from utils.debug_tools import DebugTools


class CanvasDetector:
    """
    Класс для автоматического обнаружения холста и расчета его сетки.
    """

    def __init__(self, capturer: ScreenCapturer, debug: bool = False):
        """
        Инициализация детектора холста.

        Args:
            capturer: Экземпляр ScreenCapturer для захвата экрана.
            debug: Включение режима отладки с OpenCV.
        """
        self.logger = logging.getLogger(__name__)
        self.capturer = capturer
        self.debug_tools = DebugTools(debug)
        self.canvas_top_left: Optional[Tuple[int, int]] = None
        self.canvas_bottom_right: Optional[Tuple[int, int]] = None
        self.cell_cols: int = 0
        self.cell_rows: int = 0
        self.logger.debug("CanvasDetector инициализирован.")

    def detect_canvas(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Автоматическое определение области холста на экране.

        Returns:
            Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
                Координаты верхнего левого и нижнего правого углов или (None, None) при ошибке.
        """
        self.logger.info("Попытка автоопределения холста...")
        try:
            # Захват полного экрана
            full_img = self.capturer.capture_fullscreen()
            if full_img is None:
                self.logger.error("Не удалось захватить экран для определения холста.")
                return None, None

            h, w, _ = full_img.shape
            # Ограничиваем область поиска нижними 70% экрана (предполагаем, что палитра выше)
            search_offset_y = int(h * 0.3)
            canvas_region = full_img[search_offset_y:h, :]

            # Преобразование в градации серого и адаптивная пороговая обработка
            gray = cv2.cvtColor(canvas_region, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            self.debug_tools.show_image("Canvas Detect - Threshold", thresh)

            # Поиск контуров
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidate_rects: List[Tuple[int, int, int, int]] = []
            debug_img = canvas_region.copy()

            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:  # Ищем прямоугольники
                    x, y, w_box, h_box = cv2.boundingRect(approx)
                    aspect_ratio = w_box / h_box
                    # Фильтр по размеру и соотношению сторон (почти квадратные ячейки)
                    if 0.8 < aspect_ratio < 1.2 and 5 < w_box < 100 and 5 < h_box < 100:
                        screen_x1 = x
                        screen_x2 = x + w_box
                        screen_y1 = y + search_offset_y
                        screen_y2 = screen_y1 + h_box
                        candidate_rects.append((screen_x1, screen_y1, screen_x2, screen_y2))
                        cv2.rectangle(debug_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

            if not candidate_rects:
                self.logger.error("Не удалось обнаружить ячейки холста.")
                self.debug_tools.show_image("Canvas Detect - No Cells", debug_img)
                return None, None

            self.debug_tools.show_image("Canvas Detect - Cells", debug_img)

            # Расчет средней ячейки
            cell_widths = [rect[2] - rect[0] for rect in candidate_rects]
            cell_heights = [rect[3] - rect[1] for rect in candidate_rects]
            avg_cell_width = int(np.mean(cell_widths))
            avg_cell_height = int(np.mean(cell_heights))
            self.logger.info(f"Средний размер ячейки: {avg_cell_width}x{avg_cell_height}")

            # Определение границ холста
            all_x = [rect[0] for rect in candidate_rects] + [rect[2] for rect in candidate_rects]
            all_y = [rect[1] for rect in candidate_rects] + [rect[3] for rect in candidate_rects]
            top_left = (min(all_x), min(all_y))
            bottom_right = (max(all_x), max(all_y))

            # Расчет сетки
            canvas_width = bottom_right[0] - top_left[0]
            canvas_height = bottom_right[1] - top_left[1]
            self.cell_cols = max(1, canvas_width // avg_cell_width)
            self.cell_rows = max(1, canvas_height // avg_cell_height)

            # Коррекция ориентации, если ширина больше высоты
            if self.cell_cols > self.cell_rows:
                self.cell_cols, self.cell_rows = self.cell_rows, self.cell_cols
                self.logger.info(f"Ориентация скорректирована: {self.cell_cols}x{self.cell_rows}")

            self.logger.info(f"Рассчитанная сетка холста: {self.cell_cols}x{self.cell_rows}")
            self.logger.info(f"Автоопределенный холст: TL={top_left}, BR={bottom_right}")
            return top_left, bottom_right

        except Exception as e:
            self.logger.exception(f"Ошибка автоопределения холста: {e}")
            return None, None

    def set_canvas(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> None:
        """
        Устанавливает координаты холста вручную и рассчитывает сетку.

        Args:
            top_left: Координаты верхнего левого угла.
            bottom_right: Координаты нижнего правого угла.
        """
        self.canvas_top_left = top_left
        self.canvas_bottom_right = bottom_right

        # Захват области холста для расчета сетки
        canvas_img = self.capturer.capture_area(top_left, bottom_right)
        if canvas_img is not None:
            h, w, _ = canvas_img.shape
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cell_sizes = []
            for cnt in contours:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                if 5 < w_box < 100 and 5 < h_box < 100:
                    cell_sizes.append((w_box, h_box))

            if cell_sizes:
                avg_cell_width = int(np.mean([size[0] for size in cell_sizes]))
                avg_cell_height = int(np.mean([size[1] for size in cell_sizes]))
                self.cell_cols = max(1, w // avg_cell_width)
                self.cell_rows = max(1, h // avg_cell_height)
                self.logger.info(f"Сетка рассчитана вручную: {self.cell_cols}x{self.cell_rows}")
            else:
                # Если ячейки не найдены, используем пропорциональное деление
                self.cell_cols = max(1, w // 20)  # Примерное значение
                self.cell_rows = max(1, h // 20)
                self.logger.warning(
                    f"Ячейки не найдены. Использована приблизительная сетка: {self.cell_cols}x{self.cell_rows}")

        self.logger.info(f"Холст установлен вручную: TL={top_left}, BR={bottom_right}")
