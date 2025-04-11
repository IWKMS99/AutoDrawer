import logging
import time
from typing import Optional, Dict, List, Tuple, Set
import numpy as np
from tqdm import tqdm

from core.state_manager import StateManager
from capture.canvas_detector import CanvasDetector
from capture.palette_analyzer import PaletteAnalyzer
from image_processing.image_processor import ImageProcessor
from image_processing.color_matcher import ColorMatcher
from input.mouse_controller import MouseController
from input.input_validator import InputValidator
from utils.config_loader import ConfigLoader
from utils.debug_tools import DebugTools
from utils.helpers import optimize_drawing_order


class ProcessController:
    """
    Управляет последовательностью действий: настройка, предпросмотр, очистка, рисование.
    Делегирует задачи соответствующим модулям.
    """

    def __init__(
            self,
            state: StateManager,
            mouse: MouseController,
            canvas_detector: CanvasDetector,
            palette_analyzer: PaletteAnalyzer,
            image_processor: ImageProcessor,
            color_matcher: ColorMatcher,
            config: ConfigLoader,
            debug: bool
    ):
        """
        Инициализация контроллера процесса.

        Args:
            state: Управление состоянием программы.
            mouse: Управление мышью.
            canvas_detector: Обнаружение холста.
            palette_analyzer: Анализ палитры.
            image_processor: Обработка изображений.
            color_matcher: Сопоставление цветов.
            config: Конфигурация настроек.
            debug: Включение режима отладки.
        """
        self.logger = logging.getLogger(__name__)
        self.state = state
        self.mouse = mouse
        self.canvas_detector = canvas_detector
        self.palette_analyzer = palette_analyzer
        self.image_processor = image_processor
        self.color_matcher = color_matcher
        self.config = config
        self.debug_tools = DebugTools(debug)
        self.validator = InputValidator()

        # Параметры, устанавливаемые в setup
        self.num_colors: Optional[int] = None
        self.target_resolution_w: Optional[int] = None
        self.target_resolution_h: Optional[int] = None

        self.logger.debug("ProcessController инициализирован.")

    def setup(self, clear_requested: bool) -> bool:
        """
        Настройка всех параметров: холст, палитра, параметры изображения, скорость.

        Args:
            clear_requested: Нужно ли очищать холст.

        Returns:
            bool: True, если настройка успешна, иначе False.
        """
        try:
            print("\n--- Начало настройки ---")
            print("Нажмите ESC для отмены в любой момент.")
            print("Используйте клики левой кнопкой мыши для выбора областей.")

            # 1. Настройка холста
            if not self.setup_canvas(clear_requested):
                return False

            # 2. Настройка палитры
            if not self.setup_palette(clear_requested):
                return False

            # 3. Настройка параметров
            if not self.setup_parameters():
                return False

            return True

        except KeyboardInterrupt:
            self.logger.warning("Настройка прервана пользователем.")
            self.state.stop()
            return False
        except Exception as e:
            self.logger.exception(f"Ошибка настройки: {e}")
            self.state.stop()
            return False

    def setup_canvas(self, clear_requested: bool) -> bool:
        """Настройка области холста."""
        self.logger.info("Настройка области холста...")
        print("\n1. Определение области холста...")

        canvas_tl, canvas_br = self.canvas_detector.detect_canvas()
        if canvas_tl and canvas_br:
            self.logger.info("Холст определен автоматически.")
            self.canvas_detector.set_canvas(canvas_tl, canvas_br)
        else:
            self.logger.warning("Автоопределение холста не удалось. Запрашиваем вручную.")
            canvas_tl, canvas_br = self.mouse.get_area(
                "Выберите ОБЛАСТЬ ХОЛСТА для рисования (кликните верхний левый, затем нижний правый угол)."
            )
            if not canvas_tl or not self.state.is_running():
                self.logger.warning("Выбор области холста отменен.")
                self.state.stop()
                return False
            self.canvas_detector.set_canvas(canvas_tl, canvas_br)

        self.logger.info(
            f"Холст: TL={canvas_tl}, BR={canvas_br}, Сетка={self.canvas_detector.cell_cols}x{self.canvas_detector.cell_rows}")
        return True

    def setup_palette(self, clear_requested: bool) -> bool:
        """Настройка палитры и ластика."""
        self.logger.info("Настройка палитры...")
        print("\n2. Определение области палитры...")

        palette_tl, palette_br, circles, eraser_pos = self.palette_analyzer.detect_palette()
        if palette_tl and palette_br:
            self.logger.info("Палитра определена автоматически.")
            self.palette_analyzer.set_palette(palette_tl, palette_br, circles)
            if clear_requested and eraser_pos:
                self.palette_analyzer.set_eraser(eraser_pos)
                self.logger.info(f"Ластик найден автоматически: {eraser_pos}")
            elif clear_requested:
                self.logger.warning("Ластик не найден автоматически. Запрашиваем вручную.")
                print("--> Кликните на инструмент ЛАСТИК на палитре.")
                eraser_pos = self.mouse.get_click("(Ластик)")
                if not eraser_pos or not self.state.is_running():
                    self.logger.warning("Выбор ластика отменен.")
                    self.state.stop()
                    return False
                self.palette_analyzer.set_eraser(eraser_pos)
        else:
            self.logger.warning("Автоопределение палитры не удалось. Запрашиваем вручную.")
            palette_tl, palette_br = self.mouse.get_area(
                "Выберите ОБЛАСТЬ ПАЛИТРЫ с цветами (кликните верхний левый, затем нижний правый угол)."
            )
            if not palette_tl or not self.state.is_running():
                self.logger.warning("Выбор области палитры отменен.")
                self.state.stop()
                return False
            self.palette_analyzer.set_palette(palette_tl, palette_br, [])
            if clear_requested:
                print("--> Кликните на инструмент ЛАСТИК на палитре.")
                eraser_pos = self.mouse.get_click("(Ластик)")
                if not eraser_pos or not self.state.is_running():
                    self.logger.warning("Выбор ластика отменен.")
                    self.state.stop()
                    return False
                self.palette_analyzer.set_eraser(eraser_pos)

        self.logger.info(f"Палитра: TL={palette_tl}, BR={palette_br}")
        if eraser_pos:
            self.logger.info(f"Ластик: {eraser_pos}")
        return True

    def setup_parameters(self) -> bool:
        """Настройка параметров изображения и скорости."""
        self.logger.info("Настройка параметров...")
        print("\n3. Настройка параметров...")

        # Количество цветов
        self.num_colors = self.validator.get_positive_int(
            "Введите ОЖИДАЕМОЕ КОЛИЧЕСТВО ЦВЕТОВ в палитре",
            default=self.config.get("num_colors", 10)
        )
        if not self.state.is_running():
            return False
        self.logger.info(f"Ожидаемое количество цветов: {self.num_colors}")

        # Разрешение изображения (используем сетку холста)
        self.target_resolution_w = self.canvas_detector.cell_cols
        self.target_resolution_h = self.canvas_detector.cell_rows
        self.logger.info(f"Целевое разрешение: {self.target_resolution_w}x{self.target_resolution_h}")

        # Настройка скорости
        print("\n4. Настройка скорости...")
        self.config.update({
            "click_delay": self.validator.get_float(
                f"Задержка между кликами рисования (сек, Enter={self.config.get('click_delay', 0.5):.3f}): ",
                default=self.config.get("click_delay", 0.5),
                non_negative=True
            ),
            "color_change_delay": self.validator.get_float(
                f"Задержка при смене цвета (сек, Enter={self.config.get('color_change_delay', 0.08):.3f}): ",
                default=self.config.get("color_change_delay", 0.08),
                non_negative=True
            ),
            "clear_click_delay": self.validator.get_float(
                f"Задержка между кликами ластика (сек, Enter={self.config.get('clear_click_delay', 0.5):.3f}): ",
                default=self.config.get("clear_click_delay", 0.5),
                non_negative=True
            )
        })

        if not self.state.is_running():
            return False
        self.logger.info(
            f"Скорость: клик={self.config.get('click_delay'):.3f}с, "
            f"смена цвета={self.config.get('color_change_delay'):.3f}с, "
            f"очистка={self.config.get('clear_click_delay'):.3f}с"
        )
        return True

    def preview_image(self, input_path: str) -> bool:
        """
        Создание и показ предпросмотра изображения.

        Args:
            input_path: Путь к изображению.

        Returns:
            bool: True, если пользователь подтвердил, иначе False.
        """
        self.logger.info("Создание предпросмотра...")
        return self.image_processor.preview_image(
            input_path,
            self.target_resolution_w,
            self.target_resolution_h,
            self.state
        )

    def execute(self, input_path: str, clear_first: bool) -> None:
        """
        Выполнение основного процесса: очистка (если нужно) и рисование.

        Args:
            input_path: Путь к изображению.
            clear_first: Очищать ли холст.
        """
        self.logger.info("Запуск процесса выполнения...")

        # Пауза перед началом
        print("-" * 40)
        self.logger.info("Подготовка завершена. Переключитесь на окно для рисования!")
        self.logger.info("Начало через 5 секунд... Нажмите ПРОБЕЛ для паузы или ESC для отмены.")
        print("-" * 40)
        for i in range(5, 0, -1):
            print(f"...{i}")
            time.sleep(1)
            if not self.state.is_running():
                self.logger.warning("Запуск отменен.")
                return
            if self.state.is_paused():
                self.logger.info("Пауза активирована. Нажмите Пробел для старта.")
                print("(Пауза активна. Нажмите Пробел для старта...)")
                if not self.state.wait_while_paused():
                    return

        # Очистка холста
        if clear_first:
            self.logger.info("Очистка холста...")
            if not self.clear_canvas():
                self.logger.error("Очистка холста не удалась.")
                return

        # Рисование
        self.logger.info("Рисование изображения...")
        self.draw_image(input_path)

    def clear_canvas(self) -> bool:
        """
        Очистка холста с использованием ластика.

        Returns:
            bool: True, если очистка успешна, иначе False.
        """
        if not self.palette_analyzer.eraser_pos or not self.canvas_detector.canvas_top_left:
            self.logger.error("Не заданы параметры для очистки.")
            return False

        self.logger.info("Начало очистки холста...")
        if not self.mouse.select_color(self.palette_analyzer.eraser_pos):
            return False

        canvas_width = self.canvas_detector.canvas_bottom_right[0] - self.canvas_detector.canvas_top_left[0]
        canvas_height = self.canvas_detector.canvas_bottom_right[1] - self.canvas_detector.canvas_top_left[1]
        total_clicks = self.canvas_detector.cell_cols * self.canvas_detector.cell_rows

        self.logger.info(f"Очистка {self.canvas_detector.cell_cols}x{self.canvas_detector.cell_rows} клеток...")
        with tqdm(total=total_clicks, desc="Очистка холста", unit="click") as pbar:
            for y in range(self.canvas_detector.cell_rows):
                for x in range(self.canvas_detector.cell_cols):
                    if not self.state.wait_while_paused():
                        self.logger.warning("Очистка прервана.")
                        return False

                    step_x = canvas_width / self.canvas_detector.cell_cols
                    step_y = canvas_height / self.canvas_detector.cell_rows
                    screen_x = int(self.canvas_detector.canvas_top_left[0] + x * step_x + step_x / 2)
                    screen_y = int(self.canvas_detector.canvas_top_left[1] + y * step_y + step_y / 2)

                    screen_x = min(max(screen_x, self.canvas_detector.canvas_top_left[0]),
                                   self.canvas_detector.canvas_bottom_right[0] - 1)
                    screen_y = min(max(screen_y, self.canvas_detector.canvas_top_left[1]),
                                   self.canvas_detector.canvas_bottom_right[1] - 1)

                    if not self.mouse.click(screen_x, screen_y, delay=self.config.get("clear_click_delay")):
                        self.logger.error("Ошибка при очистке пикселя.")
                        return False
                    pbar.update(1)

        self.logger.info("Очистка холста завершена.")
        return True

    def draw_image(self, input_path: str) -> None:
        """
        Рисование изображения на холсте.

        Args:
            input_path: Путь к изображению.
        """
        if not self.canvas_detector.canvas_top_left or not self.canvas_detector.canvas_bottom_right:
            self.logger.error("Область холста не определена.")
            return

        # Загрузка пикселей изображения
        pixels, img_w, img_h = self.image_processor.get_input_pixels(
            input_path,
            self.target_resolution_w,
            self.target_resolution_h,
            self.num_colors
        )
        if not pixels:
            self.logger.error("Не удалось загрузить пиксели изображения.")
            return

        # Захват палитры
        palette_img = self.palette_analyzer.capture_palette()
        if palette_img is None:
            self.logger.error("Не удалось захватить палитру.")
            self.state.stop()  # Останавливаем программу при ошибке
            return

        # Извлечение цветов палитры
        palette_data = self.palette_analyzer.extract_colors(palette_img, self.num_colors)
        if not palette_data:
            self.logger.error("Не удалось извлечь цвета палитры.")
            return
        self.logger.info(f"Извлечено {len(palette_data)} цветов палитры.")

        # Сопоставление цветов
        unique_colors: Set[Tuple[int, int, int]] = set(p[2] for p in pixels)
        color_map = self.color_matcher.map_colors(unique_colors, palette_data)
        if not color_map:
            self.logger.error("Не удалось сопоставить цвета.")
            return

        # Оптимизация порядка рисования
        drawable_pixels = [p for p in pixels if p[2] in color_map]
        optimized_pixels = optimize_drawing_order(drawable_pixels)
        self.logger.info(f"Будет нарисовано {len(optimized_pixels)} из {len(pixels)} пикселей.")

        # Рисование
        canvas_width = self.canvas_detector.canvas_bottom_right[0] - self.canvas_detector.canvas_top_left[0]
        canvas_height = self.canvas_detector.canvas_bottom_right[1] - self.canvas_detector.canvas_top_left[1]
        step_x = canvas_width / self.canvas_detector.cell_cols
        step_y = canvas_height / self.canvas_detector.cell_rows

        with tqdm(total=len(optimized_pixels), desc="Рисование", unit="px") as pbar:
            for x, y, color in optimized_pixels:
                if not self.state.wait_while_paused():
                    self.logger.warning("Рисование прервано.")
                    return

                palette_pos = color_map.get(color)
                if not palette_pos:
                    pbar.update(1)
                    continue

                if not self.mouse.select_color(palette_pos):
                    self.logger.error("Ошибка выбора цвета.")
                    return

                screen_x = int(self.canvas_detector.canvas_top_left[0] + x * step_x + step_x / 2)
                screen_y = int(self.canvas_detector.canvas_top_left[1] + y * step_y + step_y / 2)
                screen_x = min(max(screen_x, self.canvas_detector.canvas_top_left[0]),
                               self.canvas_detector.canvas_bottom_right[0] - 1)
                screen_y = min(max(screen_y, self.canvas_detector.canvas_top_left[1]),
                               self.canvas_detector.canvas_bottom_right[1] - 1)

                if not self.mouse.click(screen_x, screen_y, delay=self.config.get("click_delay")):
                    self.logger.error("Ошибка рисования пикселя.")
                    return
                pbar.update(1)

        self.logger.info("Рисование завершено.")
