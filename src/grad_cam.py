import numpy as np
import matplotlib.pyplot as plt    # визуализация графиков и изображений
import math
import re
from pathlib import Path           # удобная работа с путями (совместима с Windows/Linux)
from PIL import Image              # загрузка и обработка изображений (PNG, JPEG)
import torch.nn as nn

# обертка: GradCAM вызывает forward(x_img), мета подставляется внутрь
class _CamWrapper(nn.Module):
    '''
    Обертка над моделью, ожидающей (x_img, x_meta), чтобы Grad-CAM мог
    вызывать forward(x_img). Метаданные задаются через set_meta().
    '''
    def __init__(self, base_model):
        '''
        Инициализирует обертку базовой моделью.

        Args:
            base_model (nn.Module): Модель, ожидающая (x_img, x_meta).
        '''
        super().__init__()
        self.base_model = base_model
        # Шаг 1. Хранилище для мета‑признаков, которые подставим в forward()
        self._x_meta = None
    def set_meta(self, x_meta):
        '''
        Сохраняет тензор метаданных для использования в forward.

        Args:
            x_meta (torch.Tensor): Тензор метаданных формы (B, F) или (1, F).
        '''
        # Шаг 2. Сохраняем мета‑признаки для последующего вызова forward(x_img)
        self._x_meta = x_meta
    def forward(self, x_img):
        '''
        Прямой проход: добавляет к изображениям сохраненные метаданные.

        Args:
            x_img (torch.Tensor): Батч изображений (B, C, H, W).

        Returns:
            torch.Tensor: Выход базовой модели.
        '''
        # Расширяем x_meta под размер батча и передаем в базовую модель
        return self.base_model(x_img, self._x_meta.expand(x_img.size(0), -1))
    
def show_gradcam_png_grid(files, cols=3, id_to_label=None, figsize_unit=4):
    '''
    Показывает сетку PNG (например, Grad-CAM) с подписями true/pred.

    Если имя файла содержит шаблоны true<id> и pred<id>, под каждым
    изображением выводятся подписи; цвет зеленый при совпадении, иначе красный.

    Args:
        files (list[str|Path]): Пути к PNG-файлам.
        cols (int): Число столбцов сетки.
        id_to_label (dict|None): Сопоставление id -> текст метки.
        figsize_unit (float): Масштаб единицы размера фигуры.
    '''

    # Приводим пути к Path для унификации работы с ФС
    # Шаг 1. Нормализуем пути: приводим вход к списку Path
    files = [Path(f) for f in files]
    if not files:
        print('Нет файлов для показа.')
        return

    # Рассчитываем сетку и создаем фигуру
    # Шаг 2. Создаём сетку под изображения (rows x cols)
    rows = math.ceil(len(files) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_unit, rows * figsize_unit),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    # Отображаем изображения и отключаем оси
    # Шаг 3. Отрисовываем каждое изображение и, если возможно, подпишем true/pred
    for ax, f in zip(axes, files):
        img = Image.open(f)
        ax.imshow(img)
        ax.axis('off')

        # достаем метки из имени файла
        # Извлекаем true/pred из имени файла
        # Шаг 3.1. Пытаемся извлечь метки из имени файла
        name = f.stem
        m_true = re.search(r'true(\d+)', name)
        m_pred = re.search(r'pred(\d+)', name)
        y = int(m_true.group(1)) if m_true else None
        p = int(m_pred.group(1)) if m_pred else None

        if y is not None and p is not None:
            # Подготавливаем текст и цвет подписи
            true_txt = id_to_label.get(y, y) if id_to_label else str(y)
            pred_txt = id_to_label.get(p, p) if id_to_label else str(p)
            color = 'green' if y == p else 'red'
            ax.text(0.5, 0.02, f'true: {true_txt}\npred: {pred_txt}',
                    transform=ax.transAxes, ha='center', va='bottom',
                    color=color, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Убираем лишние пустые оси
    # Шаг 4. Глушим лишние оси
    for ax in axes[len(files):]:
        ax.axis('off')

    # Шаг 5. Показываем итоговую фигуру
    plt.show()


