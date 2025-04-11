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
    def __init__(self, debug: bool = False, monitor_idx: Optional[int] = None,
                 config_loader: Optional[ConfigLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация AutoDrawer...")

        self.debug = debug
        self.config = config_loader or ConfigLoader()
        self.state = StateManager()

        # Инициализация с учетом монитора
        self.capturer = ScreenCapturer()
        if monitor_idx is not None:
            if not self.capturer.set_monitor(monitor_idx):
                self.logger.error("Неверный индекс монитора. Используется монитор по умолчанию (0).")
        self.canvas_detector = CanvasDetector(self.capturer, debug, config_loader=self.config)
        self.palette_analyzer = PaletteAnalyzer(self.capturer, debug)

        self.image_processor = ImageProcessor(debug)
        self.color_matcher = ColorMatcher()
        self.mouse = MouseController(self.state, self.config)
        self.keyboard = KeyboardListener(self.state)
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
        self.debug_tools = DebugTools(debug)
        self.keyboard.start()
        self.logger.info("AutoDrawer успешно инициализирован.")

    def setup(self, clear_requested: bool) -> bool:
        self.logger.info("--- Начало настройки AutoDrawer ---")
        try:
            # Выбор монитора, если не задан
            if self.capturer.selected_monitor_idx is None:
                print("\nДоступные мониторы:")
                for i, mon in enumerate(self.capturer.monitors):
                    print(f"{i}: {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})")
                monitor_idx = int(input("Выберите номер монитора (Enter для 0): ") or 0)
                if not self.capturer.set_monitor(monitor_idx):
                    self.logger.warning("Неверный выбор монитора. Используется монитор 0.")
                    self.capturer.set_monitor(0)

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
        start_time = time.time()
        self.logger.info("=" * 40)
        self.logger.info(f"Запуск AutoDrawer: {input_path}, очистка: {clear_first}")
        self.logger.info("=" * 40)

        try:
            if not self.setup(clear_first):
                self.logger.warning("Программа остановлена из-за ошибки настройки.")
                return

            if not self.state.is_running():
                self.logger.warning("Программа остановлена после настройки.")
                return

            if not self.process.preview_image(input_path):
                self.logger.info("Рисование отменено после предпросмотра.")
                return

            if not self.state.is_running():
                self.logger.warning("Программа остановлена после предпросмотра.")
                return

            self.process.execute(input_path, clear_first)

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
            self.keyboard.stop()
            if self.debug:
                self.debug_tools.cleanup()
            end_time = time.time()
            self.logger.info(f"Общее время выполнения: {end_time - start_time:.2f} секунд.")
            self.logger.info("Завершение работы AutoDrawer.")

    def __del__(self):
        self.keyboard.stop()
        if self.debug:
            self.debug_tools.cleanup()
        self.logger.debug("AutoDrawer удален.")
