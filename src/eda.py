import numpy as np
import pandas as pd
import re                           # регулярные выражения (для извлечения текста, чисел и пр.)
import ast                          # безопасное преобразование строк в объекты Python (например, списки)
import pydicom                      # чтение и обработка медицинских DICOM-файлов
import matplotlib.pyplot as plt     # построение графиков и визуализация изображений
from pathlib import Path            # удобная работа с путями и файловой системой
from PIL import Image               # открытие, обработка и сохранение изображений (PNG, JPEG и т.п.)
from tqdm import tqdm               # прогресс-бары при обработке большого количества файлов
from pydicom.pixel_data_handlers.util import apply_voi_lut  # применение LUT для корректной яркости и контраста

def convert_dicom_folder_to_png(input_dir, output_dir):
    # Шаги выполнения:
    # 1) Нормализуем пути и создаём папку назначения
    # 2) Обходим все .dcm файлы и читаем DICOM с учётом VOI LUT
    # 3) Обрабатываем фотометрию MONOCHROME1 (инвертируем градации)
    # 4) Нормируем массив в [0..1] и сохраняем PNG в 8‑битном диапазоне
    # 5) Пропускаем файлы с ошибками, выводим краткое сообщение
    '''
    Конвертирует все DICOM-файлы в папке в PNG-изображения.

    Для каждого .dcm применяется VOI LUT (если есть), выполняется инверсия
    для MONOCHROME1, нормализация в диапазон 0..255 и сохранение в PNG.

    Args:
        input_dir (str | Path): Путь к папке с .dcm файлами.
        output_dir (str | Path): Папка для сохранения .png. Будет создана.
    '''

    # Приводим пути к Path и гарантируем существование выходной папки
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Итерируемся по DICOM-файлам с прогресс-баром
    for dcm_path in tqdm(sorted(input_dir.glob('*.dcm')), desc=f'Converting {input_dir.name}'):
        out_path = output_dir / f'{dcm_path.stem}.png'
        if out_path.exists():
            continue  # не перезаписываем
        try:
            # Читаем DICOM (force=True на случай частично некорректных заголовков)
            ds = pydicom.dcmread(dcm_path, force=True)

            # применяем LUT если доступно
            try:
                arr = apply_voi_lut(ds.pixel_array, ds)
            except Exception:
                arr = ds.pixel_array

            # инверсия, если MONOCHROME1
            if getattr(ds, 'PhotometricInterpretation', '').upper() == 'MONOCHROME1':
                # Инвертируем яркости для MONOCHROME1 (черное/белое наоборот)
                arr = arr.max() - arr

            # Нормализация в диапазон 0..255 с небольшой эпсилон-защитой
            arr = arr.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5)
            arr = (arr * 255).astype(np.uint8)

            # Сохраняем массив как PNG-изображение
            Image.fromarray(arr).save(out_path)

        except Exception as e:
            print(f'[skip] {dcm_path.name}: {e}')

    print(f'Готово: {len(list(output_dir.glob('*.png')))} PNG сохранено в {output_dir}')

def show_images_examples(png_location, df_classes, classes, samples_per_class=2, seed=42):
    # Шаги выполнения:
    # 1) Создать сетку под визуализацию (строки — классы, столбцы — примеры)
    # 2) Для каждого класса выбрать несколько patientId и отрисовать PNG
    # 3) Подписать класс и усечённый ID; выключить оси
    # 4) Показать итоговую фигуру
    '''
    Показывает примеры изображений по классам.

    Берет по N примеров из df_classes для каждого целевого класса,
    загружает PNG по patientId и строит сетку с подписями.

    Args:
        png_location (str | Path): Папка с PNG-файлами (имена = patientId).
        df_classes (pd.DataFrame): Таблица с колонками 'patientId', 'class'.
        samples_per_class (int): Число примеров на класс.
        seed (int): Зерно для воспроизводимости семплирования.
    '''

    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(12, 8))
    for i, cls in enumerate(classes):
        # Отбираем записи текущего класса
        subset = df_classes[df_classes['class'] == cls]
        sampled = subset.sample(samples_per_class, random_state=seed)

        for j, (_, row) in enumerate(sampled.iterrows()):
            pid = row['patientId']
            img_path = png_location / f'{pid}.png'

            if not img_path.exists():
                continue

            # Читаем изображение и приводим к оттенкам серого
            img = Image.open(img_path).convert('L')
            ax = axes[i, j]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{cls}\nID: {pid[:8]}...', fontsize=9)
            ax.axis('off')

    plt.suptitle('Примеры снимков по категориям', fontsize=14)
    plt.tight_layout()
    plt.show()

def parse_dicom_age(value):
    # Шаги выполнения:
    # 1) Привести значение к строке и убрать пробелы
    # 2) Распарсить шаблон NNNU, где U ∈ {Y,M,W,D}
    # 3) Перевести в годы (float) и проверить разумные границы
    '''
    Парсит DICOM-возраст в годы.

    Поддерживает единицы Y/M/W/D, возвращает float лет или None
    при некорректном формате/диапазоне.

    Args:
        value (str | int): Значение возраста, например '045Y'.

    Returns:
        float | None: Возраст в годах.
    '''

    s = str(value).strip()
    # Ищем число и необязательную единицу (Y/M/W/D)
    m = re.fullmatch(r'(?i)(\d{1,3})([YMWD])?', s)  # допускаем отсутствие суффикса -> годы
    if not m:
        return None
    n = int(m.group(1))
    u = (m.group(2) or 'Y').upper()
    # Переводим в годы в зависимости от единицы
    years = {'Y': n, 'M': n/12.0, 'W': n/52.0, 'D': n/365.0}.get(u, n)
    if years < 0 or years > 120:
        return None
    return float(years)

def dicom_meta_to_csv(dcm_location='', fields=[]):
    # Шаги выполнения:
    # 1) Собрать список .dcm файлов из папки
    # 2) Для каждого файла прочитать DICOM и извлечь ключевые теги
    # 3) Сложить строки в список и вернуть DataFrame
    '''
    Извлекает метаданные из DICOM-файлов в DataFrame.

    Проходит по всем .dcm в указанной папке, считывает ключевые поля
    (возраст, пол, поза, размеры, PixelSpacing) и возвращает таблицу.
    Параметр fields зарезервирован (пока не используется).

    Args:
        dcm_location (str | Path): Путь к папке с DICOM.
        fields (list): Зарезервировано для выбора полей.

    Returns:
        pd.DataFrame: Таблица с колонками patientId, sex, age, view_position,
        rows, cols, pixel_spacing.
    '''

    dcm_dir = Path(dcm_location)
    dcm_files = sorted(dcm_dir.glob('*.dcm'))
    meta_data = []

    # Проходим по каждому .dcm с прогресс-баром
    for dcm_path in tqdm(dcm_files, desc='Парсим DICOM метаинформацию'):
        try:
            ds = pydicom.dcmread(dcm_path, force=True)
            pid = dcm_path.stem

            age = parse_dicom_age(getattr(ds, 'PatientAge', ''))
            sex = getattr(ds, 'PatientSex', None) or None
            view_pos = getattr(ds, 'ViewPosition', None) or None
            rows = getattr(ds, 'Rows', None)
            cols = getattr(ds, 'Columns', None)
            pixel_spacing = getattr(ds, 'PixelSpacing', None)

            meta_data.append({
                'patientId': pid,
                'sex': sex,
                'age': age,
                'view_position': view_pos,
                'rows': rows,
                'cols': cols,
                'pixel_spacing': str(pixel_spacing) if pixel_spacing is not None else 'None'
            })
        except Exception as e:
            print(f'[WARN] {dcm_path.name}: {e}')

    return pd.DataFrame(meta_data)

def parse_spacing(value):
    # Шаги выполнения:
    # 1) Попытаться разобрать строку (например, '[0.7, 0.7]') в список/кортеж
    # 2) Если формат валиден, вернуть пару (float, float)
    # 3) Иначе вернуть (None, None)
    '''
    Парсит PixelSpacing из строки в пару чисел.

    Ожидает строку-литерал списка/кортежа (например '[0.7, 0.7]') и
    возвращает (float, float) либо (None, None) при ошибке.

    Args:
        value (str): Строковое представление списка/кортежа.

    Returns:
        tuple[float | None, float | None]: Пиксельные шаги по осям.
    '''

    try:
        spacing = ast.literal_eval(value)
        if isinstance(spacing, (list, tuple)) and len(spacing) == 2:
            return float(spacing[0]), float(spacing[1])
    except Exception:
        pass
    return None, None

