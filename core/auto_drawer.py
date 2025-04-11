import logging
import time
from typing import Optional

from capture.screen_capturer import ScreenCapturer
from capture.canvas_detector import CanvasDetector
from capture.palette_analyzer import PaletteAnalyzer
from image_processing.image_processor import ImageProcessor
from image_processing.color_matcher import ColorMatcher
from input.mouse_controller import MouseController
from input.keyboard_listener import KeyboardListener
from core.state_manager import StateManager
from core.process_controller import ProcessController
from utils.config_loader import ConfigLoader
from utils.debug_tools import DebugTools


class AutoDrawer:
    """
    Главный класс для координации процесса автоматического рисования изображения.
    Инициализирует модули и управляет основным процессом.
    """

    def __init__(self, debug: bool = False):
        """
        Инициализация AutoDrawer с настройкой всех модулей.

        Args:
            debug: Включает режим отладки с выводом OpenCV.
        """
        # Инициализация логирования
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация AutoDrawer...")

        # Флаг дебаггинга
        self.debug = debug

        # Загрузка конфигурации
        self.config = ConfigLoader()

        # Управление состоянием
        self.state = StateManager()

        # Модули захвата экрана
        self.capturer = ScreenCapturer()
        self.canvas_detector = CanvasDetector(self.capturer, debug)
        self.palette_analyzer = PaletteAnalyzer(self.capturer, debug)

        # Модули обработки изображений
        self.image_processor = ImageProcessor(debug)
        self.color_matcher = ColorMatcher()

        # Модули ввода
        self.mouse = MouseController(self.state, self.config)
        self.keyboard = KeyboardListener(self.state)

        # Контроллер процесса
        self.process = ProcessController(
            state=self.state,
            mouse=self.mouse,
            canvas_detector=self.canvas_detector,
            palette_analyzer=self.palette_analyzer,
            image_processor=self.image_processor,
            color_matcher=self.color_matcher,
            config=self.config,
            debug=debug
        )

        # Инструменты дебаггинга
        self.debug_tools = DebugTools(debug)

        # Запуск слушателя клавиатуры
        self.keyboard.start()
        self.logger.info("AutoDrawer успешно инициализирован.")

    def setup(self, clear_requested: bool) -> bool:
        """
        Настройка параметров: области холста, палитры, параметров изображения и скорости.

        Args:
            clear_requested: Нужно ли очищать холст перед рисованием.

        Returns:
            bool: True, если настройка успешна, иначе False.
        """
        self.logger.info("--- Начало настройки AutoDrawer ---")
        try:
            # Делегируем настройку контроллеру процесса
            success = self.process.setup(clear_requested)
            if not success:
                self.logger.warning("Настройка не была завершена.")
                self.state.stop()
                return False

            self.logger.info("--- Настройка завершена ---")
            return True

        except Exception as e:
            self.logger.exception(f"Неожиданная ошибка во время настройки: {e}")
            self.state.stop()
            return False

    def run(self, input_path: str, clear_first: bool) -> None:
        """
        Основная логика выполнения: настройка, предпросмотр, очистка и рисование.

        Args:
            input_path: Путь к входному изображению.
            clear_first: Очищать ли холст перед рисованием.
        """
        start_time = time.time()
        self.logger.info("=" * 40)
        self.logger.info(f"Запуск AutoDrawer: {input_path}, очистка: {clear_first}")
        self.logger.info("=" * 40)

        try:
            # 1. Настройка
            if not self.setup(clear_first):
                self.logger.warning("Программа остановлена из-за ошибки настройки.")
                return

            if not self.state.is_running():
                self.logger.warning("Программа остановлена после настройки.")
                return

            # 2. Предпросмотр и подтверждение
            if not self.process.preview_image(input_path):
                self.logger.info("Рисование отменено после предпросмотра.")
                return

            if not self.state.is_running():
                self.logger.warning("Программа остановлена после предпросмотра.")
                return

            # 3. Выполнение основного процесса (очистка и рисование)
            self.process.execute(input_path, clear_first)

            # Финальное сообщение
            if self.state.is_running():
                self.logger.info("=" * 40)
                self.logger.info("Рисование успешно завершено!")
                self.logger.info("=" * 40)
            else:
                self.logger.warning("=" * 40)
                self.logger.warning("Рисование было прервано.")
                self.logger.warning("=" * 40)

        except KeyboardInterrupt:
            self.logger.info("Программа прервана пользователем (Ctrl+C)")
            self.state.stop()
        except Exception as e:
            self.logger.exception(f"Критическая ошибка в run(): {e}")
            self.state.stop()
        finally:
            # Очистка ресурсов
            self.keyboard.stop()
            if self.debug:
                self.debug_tools.cleanup()
            end_time = time.time()
            self.logger.info(f"Общее время выполнения: {end_time - start_time: .2f} секунд.")
            self.logger.info("Завершение работы AutoDrawer.")

    def __del__(self):
        """Очистка ресурсов при удалении объекта."""
        self.keyboard.stop()
        if self.debug:
            self.debug_tools.cleanup()
        self.logger.debug("AutoDrawer удален.")
