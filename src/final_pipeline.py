import json
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from tqdm.auto import tqdm  # прогресс‑бар для ноутбуков и консоли
# src
from src.eda import parse_dicom_age
from src.grad_cam import _CamWrapper
from src.models import ResNet50WithMeta

def _load_best_params(reports_dir):
    '''
    Загрузка лучших гиперпараметров из reports/.

    Сначала пробуем 'best_params.json', затем 'best_optuna_params.json'.
    Если в файле есть ключ 'best_params' — разворачиваем его.
    '''
    # Список возможных файлов с лучшими гиперпараметрами (поддерживаем оба имени)
    candidates = [reports_dir / 'best_params.json', reports_dir / 'best_optuna_params.json']
    for p in candidates:
        if p.exists():
            # Читаем JSON и, если есть ключ 'best_params', возвращаем только его содержимое
            payload = json.loads(p.read_text(encoding='utf-8'))
            return payload.get('best_params', payload)
    return {}

def _load_meta_norm(reports_dir):
    '''
    Возвращает (mean, std) как pandas.Series из reports/meta_norm.json.
    '''
    # 1) Путь к JSON с параметрами нормировки метаданных
    meta_path = reports_dir / 'meta_norm.json'
    # 2) Загружаем словари mean/std и превращаем их в pandas.Series
    payload = json.loads(meta_path.read_text(encoding='utf-8'))
    # 3) Преобразуем словари в Series (удобно индексировать по именам колонок)
    mean = pd.Series(payload.get('mean', {}))
    std = pd.Series(payload.get('std', {}))
    return mean, std

def _load_model_accuracy_percent(reports_dir):
    '''
    Читает последнюю строку accuracy из reports/final_metrics.csv
    и возвращает точность в процентах (0..100). Если файла нет — None.
    '''
    # Путь к CSV со сводными метриками (accuracy, f1 и т.д.)
    csv_path = reports_dir / 'final_metrics.csv'
    if not csv_path.exists():
        return None
    try:
        # Читаем CSV и берем последнюю доступную точность
        df = pd.read_csv(csv_path)
        if 'accuracy' in df.columns and len(df) > 0:
            return float(df['accuracy'].iloc[-1] * 100.0)
    except Exception:
        # Если не удалось прочитать/сконвертировать — просто не добавляем метрику
        pass
    return None

class Pipeline:
    def __init__(self, model, meta_mean, meta_std, meta_cols_cont, meta_cols_bin, device='cuda', img_size=512):
        '''
        Компактный инференс‑пайплайн: сборка мета‑признаков, предсказание и Grad‑CAM.

        Args:
            model: итоговая модель (ResNet50WithMeta).
            meta_mean: средние значения непрерывных мета‑признаков (pd.Series).
            meta_std: ст. отклонения для мета‑признаков (pd.Series).
            meta_cols_cont: список непрерывных мета‑признаков.
            meta_cols_bin: список бинарных мета‑признаков (one‑hot).
            device: 'cuda' или 'cpu'.
            img_size: размер изображения для Resize.
        '''
        # 1) Выбираем устройство и переводим модель в режим eval на нужный девайс
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

        # 2) Подготавливаем параметры нормировки мета‑признаков
        mean = meta_mean.reindex(meta_cols_cont)
        std = meta_std.reindex(meta_cols_cont).replace(0, 1.0)  # защита от деления на 0
        self.meta_mean = torch.tensor(mean.values, dtype=torch.float32)
        self.meta_std = torch.tensor(std.values, dtype=torch.float32)
        self.meta_cols_cont = list(meta_cols_cont)
        self.meta_cols_bin = list(meta_cols_bin)

        # 3) Трансформации изображений для инференса
        self.tfm = transforms.Compose([
            transforms.Grayscale(1),                # преобразуем в 1‑канальное изображение
            transforms.Resize((img_size, img_size)),# приводим к размеру обучения из best_params
            transforms.ToTensor()                   # конвертируем в тензор [0..1]
        ])
    
    @classmethod
    def from_artifacts(cls, models_dir='models', reports_dir='reports', meta_cols_cont=None, meta_cols_bin=None, device=None):
        '''
        Сборка пайплайна из артефактов проекта.

        - Грузит models/resnet50_final.pth
        - Читает reports/best_params.json (или best_optuna_params.json) и reports/meta_norm.json
        - Настраивает размер изображения и нормировку метаданных
        '''

        # 1) Нормализуем пути к артефактам
        models_dir = Path(models_dir)
        reports_dir = Path(reports_dir)

        # 2) Загружаем лучшие гиперпараметры (например, img_size)
        best = _load_best_params(reports_dir)
        img_size = int(best.get('img_size', 512))

        # 3) Колонки метаданных по умолчанию (совместимо с проектом)
        meta_cols_cont = meta_cols_cont or ['age', 'spacing_x', 'spacing_y']
        meta_cols_bin = meta_cols_bin or ['sex_M', 'vp_AP']

        # 4) Загружаем нормировки метаданных
        meta_mean, meta_std = _load_meta_norm(reports_dir)

        # 5) Создаем модель с учетом размерности метаданных
        meta_dim = len(meta_cols_cont) + len(meta_cols_bin)
        model = ResNet50WithMeta(n_classes=3, meta_dim=meta_dim, freeze_backbone=False)

        # 6) Выбираем устройство (если не указано явно)
        resolved_device = device
        if resolved_device is None:
            resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        map_loc = torch.device(resolved_device)

        # 7) Загружаем финальные веса модели и переводим в eval
        weights_path = models_dir / 'resnet50_final.pth'
        state = torch.load(weights_path, map_location=map_loc)
        model.load_state_dict(state)
        model.eval()

        return cls(
            model,
            meta_mean,
            meta_std,
            meta_cols_cont,
            meta_cols_bin,
            device=resolved_device,
            img_size=img_size,
        )

    def read_dicom(self, path):
        # 1) Читаем DICOM (force=True — на случай неполных тегов)
        ds = pydicom.dcmread(path, force=True)
        try:
            # 2) Пытаемся применить VOI LUT (если есть в теге); иначе используем исходный массив
            arr = apply_voi_lut(ds.pixel_array, ds)
        except Exception:
            arr = ds.pixel_array
        # 3) Инвертируем изображение для MONOCHROME1 (в DICOM это «инвертированная» шкала серого)
        if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
            arr = arr.max() - arr
        # 4) Приводим к float32 и нормируем в [0..1], затем конвертируем в 8‑битный PNG образ через PIL
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        img = Image.fromarray((arr * 255).astype(np.uint8))
        return img, ds

    def prepare_meta(self, ds):
        # 1) Достаем интересующие теги из DICOM: возраст, пол, проекцию, PixelSpacing
        age = parse_dicom_age(getattr(ds, 'PatientAge', '')) or 0.0
        sex_raw = str(getattr(ds, 'PatientSex', '') or '').upper()
        vp_raw = str(getattr(ds, 'ViewPosition', '') or '').upper()
        ps = getattr(ds, 'PixelSpacing', None)
        spacing_x = float(ps[0]) if ps is not None else 0.15
        spacing_y = float(ps[1]) if ps is not None else 0.15

        # 2) Строим словарь мета‑признаков (one‑hot по полу и проекции)
        meta = {
            'age': age,
            'spacing_x': spacing_x,
            'spacing_y': spacing_y,
            'sex_M': 1.0 if sex_raw == 'M' else 0.0,
            'sex_F': 1.0 if sex_raw == 'F' else 0.0,
            'vp_AP': 1.0 if vp_raw == 'AP' else 0.0,
            'vp_PA': 1.0 if vp_raw == 'PA' else 0.0,
        }

        # 3) Формируем тензоры: отдельно непрерывные мета‑признаки (нормируем) и бинарные, затем склеиваем
        x_cont = torch.tensor([float(meta.get(c, 0.0)) for c in self.meta_cols_cont], dtype=torch.float32)
        x_cont = (x_cont - self.meta_mean.cpu()) / self.meta_std.cpu()
        x_bin = torch.tensor([float(meta.get(c, 0.0)) for c in self.meta_cols_bin], dtype=torch.float32)
        x_meta = torch.cat([x_cont, x_bin]).unsqueeze(0).to(self.device)
        return x_meta, meta

    def predict(self, dicom_path):
        # 1) Читаем DICOM и подготавливаем изображение и метаданные
        img, ds = self.read_dicom(dicom_path)
        x_img = self.tfm(img).unsqueeze(0).to(self.device)
        x_meta, meta_info = self.prepare_meta(ds)
        # 2) Инференс (с autocast на CUDA для ускорения)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            logits = self.model(x_img, x_meta)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        return pred, probs, meta_info, img

    def explain(self, x_img, x_meta, pred_cls, out_dir='reports/gradcam_infer'):
        # 1) Готовим директорию для сохранения визуализаций
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Для ResNet50 чаще всего подходит bottleneck conv3; если нет — весь блок
        # 2) Для ResNet50 обычно подойдет последний bottleneck conv3 (если нет — берем весь блок)
        target_layer = getattr(self.model.backbone.layer4[-1], 'conv3', self.model.backbone.layer4[-1])
        # 3) Оборачиваем модель, чтобы Grad‑CAM получал только x_img, а x_meta добавлялись внутри
        wrapped = _CamWrapper(self.model)
        wrapped.set_meta(x_meta)
        # 4) Считаем Grad‑CAM и накладываем тепловую карту на исходное изображение
        with GradCAM(model=wrapped, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=x_img, targets=[ClassifierOutputTarget(pred_cls)])[0]
            base = x_img[0, 0].detach().cpu().numpy()
            base = (base - base.min()) / (base.max() - base.min() + 1e-6)
            rgb = np.repeat(base[..., None], 3, axis=2)
            vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True, image_weight=0.65)
            out_path = out_dir / f'gradcam_pred_{pred_cls}.png'
            Image.fromarray(vis).save(out_path)
        return out_path

    def infer_to_json(self, dicom_path, class_names, out_json='reports/patient_inference.json'):
        '''
        Инференс одного DICOM‑файла и сохранение отчета в JSON.

        Добавляет 'model_accuracy_percent', если найден reports/final_metrics.csv.
        '''
        # 1) Инференс по одному DICOM: подготовка данных и предсказание
        # 1) Инференс по одному DICOM: подготовка данных и предсказание
        img, ds = self.read_dicom(dicom_path)
        x_img = self.tfm(img).unsqueeze(0).to(self.device)
        x_meta, meta_info = self.prepare_meta(ds)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            logits = self.model(x_img, x_meta)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

        out_json = Path(out_json)
        reports_dir = out_json.parent
        # Сохраняем Grad‑CAM рядом с reports
        # 2) Сохраняем Grad‑CAM для наглядности
        gradcam_dir = reports_dir / 'gradcam_infer'
        gradcam_path = self.explain(x_img, x_meta, pred, out_dir=gradcam_dir)

        # Вероятности в процентах (3 знака после запятой)
        # 3) Переводим вероятности в проценты (для удобства чтения отчета)
        probs_percent = {class_names[i]: round(float(probs[i]) * 100.0, 3) for i in range(len(class_names))}

        # Относительный путь от корня репозитория, если возможно
        # 4) Пытаемся сделать путь к Grad‑CAM относительным к корню репозитория
        try:
            gradcam_path_str = str(gradcam_path.relative_to(reports_dir.parent))
        except Exception:
            gradcam_path_str = str(gradcam_path)

        # 5) Формируем JSON‑отчет по пациенту
        report = {
            'patient_meta': {
                'age': meta_info.get('age'),
                'sex': 'M' if meta_info.get('sex_M', 0.0) == 1.0 else ('F' if meta_info.get('sex_F', 0.0) == 1.0 else 'Unknown'),
                'view_position': 'AP' if meta_info.get('vp_AP', 0.0) == 1.0 else ('PA' if meta_info.get('vp_PA', 0.0) == 1.0 else 'Unknown'),
                'spacing_x': meta_info.get('spacing_x'),
                'spacing_y': meta_info.get('spacing_y'),
            },
            'predicted_class': class_names[pred],
            'probabilities': probs_percent,
            'gradcam_visualization': gradcam_path_str,
        }

        # Добавляем точность модели, если доступна
        # 6) Прикладываем точность модели (если есть финальные метрики)
        acc = _load_model_accuracy_percent(reports_dir)
        if acc is not None:
            report['model_accuracy_percent'] = round(acc, 3)

        reports_dir.mkdir(parents=True, exist_ok=True)
        # 7) Сохраняем JSON‑отчет
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        return report

    def infer_dir_to_json(self, dicom_dir, class_names=None, out_json='reports/patient_inference.json', per_file=False):
        '''
        Инференс для всех файлов *.dcm в папке.

        - При per_file=True создаются отдельные JSON на каждый файл (reports/inference/<stem>.json)
        - Всегда создается агрегированный JSON (out_json) со списком items
        - Grad‑CAM сохраняется в подпапках по имени файла
        '''
        dicom_dir = Path(dicom_dir)
        out_json = Path(out_json)
        class_names = class_names or ['Normal', 'Lung Opacity', 'No Lung Opacity / Not Normal']
        reports_dir = out_json.parent
        gradcam_root = reports_dir / 'gradcam_infer'
        per_file_dir = reports_dir / 'inference'

        items = []
        dcm_files = sorted(dicom_dir.glob('*.dcm'))
        # Прогресс‑бар по количеству DICOM‑файлов
        for dcm_path in tqdm(dcm_files, desc=f'Infer {dicom_dir.name}', unit='img'):
            img, ds = self.read_dicom(dcm_path)
            x_img = self.tfm(img).unsqueeze(0).to(self.device)
            x_meta, meta_info = self.prepare_meta(ds)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                logits = self.model(x_img, x_meta)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))

            # Grad‑CAM в отдельную подпапку на файл
            subdir = gradcam_root / dcm_path.stem
            gradcam_path = self.explain(x_img, x_meta, pred, out_dir=subdir)
            try:
                gradcam_path_str = str(gradcam_path.relative_to(reports_dir.parent))
            except Exception:
                gradcam_path_str = str(gradcam_path)

            # 3) Заполняем элемент списка результатов по файлу
            probs_percent = {class_names[i]: round(float(probs[i]) * 100.0, 3) for i in range(len(class_names))}
            entry = {
                'dicom_file': dcm_path.name,
                'patient_meta': {
                    'age': meta_info.get('age'),
                    'sex': 'M' if meta_info.get('sex_M', 0.0) == 1.0 else (
                        'F' if meta_info.get('sex_F', 0.0) == 1.0 else 'Unknown'
                    ),
                    'view_position': 'AP' if meta_info.get('vp_AP', 0.0) == 1.0 else (
                        'PA' if meta_info.get('vp_PA', 0.0) == 1.0 else 'Unknown'
                    ),
                    'spacing_x': meta_info.get('spacing_x'),
                    'spacing_y': meta_info.get('spacing_y'),
                },
                'predicted_class': class_names[pred],
                'probabilities': probs_percent,
                'gradcam_visualization': gradcam_path_str,
            }

            # 4) При необходимости сохраняем отдельный JSON на файл
            if per_file:
                per_file_dir.mkdir(parents=True, exist_ok=True)
                single_out = per_file_dir / f'{dcm_path.stem}.json'
                single_report = dict(entry)
                acc = _load_model_accuracy_percent(reports_dir)
                if acc is not None:
                    single_report['model_accuracy_percent'] = round(acc, 3)
                single_out.write_text(
                    json.dumps(single_report, indent=2, ensure_ascii=False), encoding='utf-8'
                )

            items.append(entry)

        # 5) Формируем агрегированный отчет по папке
        aggregated = {
            'model_accuracy_percent': None,
            'items': items,
        }
        acc = _load_model_accuracy_percent(reports_dir)
        if acc is not None:
            aggregated['model_accuracy_percent'] = round(acc, 3)

        # 6) Сохраняем агрегированный JSON
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding='utf-8')
        return aggregated