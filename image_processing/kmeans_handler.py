import logging
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple


class KMeansHandler:
    """
    Класс для кластеризации цветов с использованием K-Means.
    """

    def __init__(self):
        """Инициализация обработчика K-Means."""
        self.logger = logging.getLogger(__name__)
        self.kmeans_handler = None  # Для хранения модели после обучения
        self.logger.debug("KMeansHandler инициализирован.")

    def cluster_colors(self, pixels: np.ndarray, num_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполняет кластеризацию цветов с помощью K-Means.

        Args:
            pixels: Массив пикселей (N, 3) в формате RGB.
            num_clusters: Количество кластеров (цветов).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Центры кластеров и метки для каждого пикселя, или (пустой массив, пустой массив) при ошибке.
        """
        self.logger.info(f"Кластеризация {pixels.shape[0]} пикселей на {num_clusters} цветов...")
        try:
            if pixels.shape[0] < num_clusters:
                self.logger.warning(f"Число пикселей ({pixels.shape[0]}) меньше числа кластеров ({num_clusters}).")
                num_clusters = max(1, pixels.shape[0])

            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(pixels)
            self.kmeans_handler = kmeans  # Сохраняем модель для возможного предсказания
            self.logger.info(f"Кластеризация завершена: {num_clusters} кластеров.")
            return kmeans.cluster_centers_, labels

        except Exception as e:
            self.logger.exception(f"Ошибка кластеризации: {e}")
            return np.array([]), np.array([])

    def predict(self, pixels: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки кластеров для новых пикселей на основе обученной модели.

        Args:
            pixels: Массив пикселей (N, 3) в формате RGB.

        Returns:
            np.ndarray: Метки кластеров или пустой массив при ошибке.
        """
        if self.kmeans_handler is None:
            self.logger.error("Модель K-Means не обучена.")
            return np.array([])

        try:
            labels = self.kmeans_handler.predict(pixels)
            self.logger.debug(f"Предсказано {len(labels)} меток.")
            return labels
        except Exception as e:
            self.logger.exception(f"Ошибка предсказания K-Means: {e}")
            return np.array([])
