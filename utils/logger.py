import logging


def setup_logger(debug: bool = False) -> None:
    """
    Настраивает глобальное логирование.

    Args:
        debug: Включает уровень DEBUG, иначе используется INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Уменьшаем шум от pynput
    logging.getLogger("pynput").setLevel(logging.WARNING)
    root_logger = logging.getLogger()
    if debug:
        root_logger.debug("Логирование настроено в режиме DEBUG.")
    else:
        root_logger.info("Логирование настроено в режиме INFO.")
