import argparse
import sys

from utils.logger import setup_logger
from core.auto_drawer import AutoDrawer
from utils.config_loader import ConfigLoader


def parse_args():
    """
    Парсит аргументы командной строки.

    Returns:
        argparse.Namespace: Объект с аргументами.
    """
    parser = argparse.ArgumentParser(description="AutoDrawer: автоматическое рисование изображений.")
    parser.add_argument("input_path", type=str, help="Путь к входному изображению.")
    parser.add_argument("--clear", action="store_true", help="Очистить холст перед рисованием.")
    parser.add_argument("--debug", action="store_true", help="Включить режим отладки с OpenCV.")
    parser.add_argument("--monitor", type=int, default=None,
                        help="Номер монитора для захвата (0 - все экраны, 1+ - конкретный).")
    return parser.parse_args()


def main():
    """Основная функция программы."""
    args = parse_args()

    # Настройка логирования
    setup_logger(debug=args.debug)

    # Инициализация конфигурации
    config_loader = ConfigLoader()

    # Инициализация и запуск AutoDrawer
    drawer = AutoDrawer(debug=args.debug, monitor_idx=args.monitor, config_loader=config_loader)
    drawer.run(args.input_path, args.clear)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем (Ctrl+C).")
        sys.exit(1)
    except Exception as e:
        print(f"\nНеожиданная ошибка: {e}")
        sys.exit(1)
