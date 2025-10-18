import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_bars(
    df_base,
    df_tr,
    labels,
    prefix,
    title,
    ylabel,
    width=0.35,
    model_names=(
        'ResNet18 (Baseline)',
        'ResNet50 (Transfer + Optuna)'
    ),
    ax=None,
):
    '''
    Сравнение метрик по классам в виде парных столбцов (baseline vs transfer).

    Ожидаются столбцы вида f'{prefix}{label}' для каждого label из labels в
    строке агрегированных метрик.

    Args:
        df_base (pd.DataFrame): метрики базовой модели.
        df_tr (pd.DataFrame): метрики transfer‑модели.
        labels (list[str]): подписи классов по X.
        prefix (str): префикс метрики (напр., 'f1_' или 'recall_').
        title (str): заголовок графика.
        ylabel (str): подпись оси Y.
        width (float): ширина столбца.
        model_names (tuple[str,str]): подписи моделей в легенде.
        ax (matplotlib.axes.Axes|None): оси, если уже созданы.

    Returns:
        matplotlib.axes.Axes: оси с отрисованным графиком.
    '''

    # Шаг 1. Готовим позиции по оси X и извлекаем значения метрик
    x = np.arange(len(labels))
    base_vals = [df_base[f'{prefix}{label}'].iat[0] for label in labels]
    tr_vals = [df_tr[f'{prefix}{label}'].iat[0] for label in labels]

    # Шаг 2. Готовим оси для отрисовки (создаём fig/ax при необходимости)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    # Шаг 3. Рисуем две группы столбцов и подписываем ось X
    ax.bar(x - width/2, base_vals, width, label=model_names[0])
    ax.bar(x + width/2, tr_vals, width, label=model_names[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)

    # Шаг 4. Оформление графика: границы, подписи, заголовок, легенда
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    # Шаг 5. Компоновка и возврат осей
    fig.tight_layout()
    return ax


def plot_metric_classes(df):
    '''
    Визуализация метрик по классам для нескольких моделей.

    Ожидается столбец 'model_pretty' и колонки с префиксами 'f1_'/'recall_'.

    Args:
        df (pd.DataFrame): таблица с метриками и подписями моделей.
    '''

    # Шаг 1. Определяем список метрик для сравнения
    metrics = [
        ('f1_', 'F1-score', 'Сравнение F1-score по классам', 'o'),
        ('recall_', 'Recall', 'Сравнение Recall по классам', 's'),
    ]

    # Шаг 2. Создаём холст и две оси (общая шкала Y)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    class_order = None

    for ax, (prefix, ylabel, title, marker) in zip(axes, metrics):
        # Шаг 3. Выбираем столбцы по префиксу и приводим к long‑формату
        cols = [
            c for c in df.columns
            if c.startswith(prefix) and not c.endswith(('macro', 'weighted', 'micro'))
        ]
        dfm = (
            df[['model_pretty'] + cols]
            .melt(id_vars='model_pretty', var_name='class', value_name='value')
        )
        dfm['class'] = dfm['class'].str.replace(prefix, '', regex=False)

        # Шаг 4. Фиксируем общий порядок классов по первой метрике
        if class_order is None:
            class_order = dfm['class'].unique().tolist()
        dfm['class'] = pd.Categorical(dfm['class'], categories=class_order, ordered=True)
        dfm = dfm.sort_values('class')

        # Шаг 5. Рисуем кривые для каждой модели
        for name, grp in dfm.groupby('model_pretty'):
            ax.plot(grp['class'], grp['value'], marker=marker, linestyle='-', label=name)

        # Шаг 6. Оформление осей, сетки и заголовка
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # Шаг 7. Легенда и компоновка всего рисунка
    axes[0].legend()
    plt.tight_layout()
    plt.show()

