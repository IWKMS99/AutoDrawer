import logging
from typing import List, Tuple


def optimize_drawing_order(pixels: List[Tuple[int, int, Tuple[int, int, int]]]) -> List[
    Tuple[int, int, Tuple[int, int, int]]]:
    """
    Оптимизирует порядок рисования пикселей для минимизации переключения цветов.

    Args:
        pixels: Список пикселей [(x, y, (r, g, b))].

    Returns:
        List[Tuple[int, int, Tuple[int, int, int]]]: Отсортированный список пикселей.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Оптимизация порядка для {len(pixels)} пикселей...")

    # Сортировка по цвету, затем по y, затем по x
    sorted_pixels = sorted(pixels, key=lambda p: (p[2], p[1], p[0]))
    logger.debug("Оптимизация порядка завершена.")
    return sorted_pixels
