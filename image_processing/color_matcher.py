import logging
import numpy as np
from typing import Set, Dict, Tuple, List
from skimage.color import rgb2lab
from scipy.spatial.distance import cdist


class ColorMatcher:
    """
    Класс для сопоставления цветов изображения с палитрой.
    """

    def __init__(self):
        """Инициализация матчера цветов."""
        self.logger = logging.getLogger(__name__)
        self.logger.debug("ColorMatcher инициализирован.")

    def map_colors(self, image_colors: Set[Tuple[int, int, int]],
                   palette_data: List[Tuple[Tuple[int, int, int], Tuple[int, int]]]) -> Dict[
        Tuple[int, int, int], Tuple[int, int]]:
        """
        Сопоставляет цвета изображения с ближайшими цветами палитры.

        Args:
            image_colors: Множество уникальных цветов изображения (RGB).
            palette_data: Список [(цвет RGB, позиция клика)] из палитры.

        Returns:
            Dict[Tuple[int, int, int], Tuple[int, int]]: Словарь {цвет изображения: позиция в палитре}.
        """
        self.logger.info(
            f"Сопоставление {len(image_colors)} цветов изображения с {len(palette_data)} цветами палитры...")
        color_map: Dict[Tuple[int, int, int], Tuple[int, int]] = {}

        if not image_colors or not palette_data:
            self.logger.warning("Пустое множество цветов или палитра.")
            return color_map

        try:
            # Извлечение цветов и позиций из palette_data
            palette_colors = np.array([color for color, _ in palette_data])
            palette_positions = [pos for _, pos in palette_data]

            # Преобразование в LAB для точного сравнения
            image_colors_array = np.array(list(image_colors), dtype=np.uint8)
            palette_colors_array = palette_colors.astype(np.uint8)

            image_lab = rgb2lab(image_colors_array.reshape(-1, 1, 3)).reshape(-1, 3)
            palette_lab = rgb2lab(palette_colors_array.reshape(-1, 1, 3)).reshape(-1, 3)

            # Вычисление расстояний между цветами
            distances = cdist(image_lab, palette_lab, metric="euclidean")
            nearest_indices = np.argmin(distances, axis=1)

            # Формирование словаря сопоставления
            for i, img_color in enumerate(image_colors):
                palette_idx = nearest_indices[i]
                color_map[img_color] = palette_positions[palette_idx]
                self.logger.debug(
                    f"Сопоставлен цвет {img_color} -> {palette_colors[palette_idx]} @ {palette_positions[palette_idx]}")

            self.logger.info(f"Сопоставлено {len(color_map)} цветов.")
            return color_map

        except Exception as e:
            self.logger.exception(f"Ошибка сопоставления цветов: {e}")
            return color_map
