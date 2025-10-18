import numpy as np                 # численные операции и работа с массивами
import torch                       # основной фреймворк PyTorch для нейронных сетей
import torch.nn as nn              # модуль для построения нейросетей (Linear, Conv, и т.д.)
from collections import Counter    # подсчет количества элементов по классам
from PIL import Image              # загрузка и обработка изображений (PNG, JPEG)
from tqdm import tqdm              # красивые прогресс-бары для циклов
from torch.utils.data import Dataset
from torchvision import models      # готовые предобученные модели (ResNet, EfficientNet и др.)
from sklearn.metrics import accuracy_score, f1_score

# Определение пользовательского датасета для задачи классификации
class RSNADataset(Dataset):
    '''
    Датасет для чтения PNG-снимков по `patientId` и меток класса.

    Конвертирует изображение в один канал ('L') и применяет трансформации,
    чтобы подготовить данные для обучения модели.
    '''
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    # Шаги выборки элемента:
    # 1) чтение PNG по patientId (grayscale 'L')
    # 2) применение transform
    # 3) извлечение метки класса
    # 4) возврат (x, y)
    def __getitem__(self, i):
        # Берем строку, читаем PNG по patientId, приводим к 'L' (1 канал),
        # применяем transform и возвращаем тензор и целевой класс.
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / f'{row.patientId}.png').convert('L') # конвертируем в grayscale (режим 'L')
        x = self.transform(img) # Трансформируем изображение в тензор
        y = int(row.class_id) # Класс легких (метка)
        # Возвращаем пару (изображение, метка)
        return x, y

class RSNADatasetWithMeta(Dataset):
    '''
    Датасет, возвращающий (x_img, x_meta, y) для моделей с табличными признаками.

    Непрерывные метапризнаки нормируются по train-статистике (mean/std),
    бинарные конкатенируются, затем массив переводится в torch.Tensor.
    '''
    def __init__(self, df, img_dir, transform, meta_cols_cont, meta_cols_bin, mean, std):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.meta_cols_cont = meta_cols_cont
        self.meta_cols_bin = meta_cols_bin
        self.mean = mean
        self.std = std

    def __len__(self): return len(self.df)
    # Шаги выборки элемента:
    # 1) чтение PNG по patientId и transform -> x_img
    # 2) сбор мета‑признаков: continuous (нормировка) + binary -> x_meta
    # 3) извлечение метки класса -> y
    # 4) возврат (x_img, x_meta, y)
    
    def __getitem__(self, i):
        # Возвращаем (тензор изображения, вектор метаданных, целевой класс).
        # Нормируем непрерывные метапризнаки и конкатенируем с бинарными.
        row = self.df.iloc[i]
        # image
        img = Image.open(self.img_dir/f'{row.patientId}.png').convert('L')
        x_img = self.transform(img)
        # Нормируем непрерывные признаки по train mean/std для стабильности
        # и объединяем с бинарными индикаторами в единый вектор

        # meta: стандартизация только континуальных, бинары как есть
        m_cont = ((row[self.meta_cols_cont].astype(float) - self.mean) / self.std).values.astype('float32')
        m_bin  = row[self.meta_cols_bin].astype('float32').values
        x_meta = np.concatenate([m_cont, m_bin]).astype('float32')
        x_meta = torch.from_numpy(x_meta)

        y = int(row.class_id)
        return x_img, x_meta, y

# Определение архитектуры нейронной сети на основе ResNet18
class ResNet18Baseline(nn.Module):
    '''
    Базовая ResNet18 для 1-канальных входов (рентген-снимки).

    Меняет первый сверточный слой на 1-канальный и переводит weights из RGB
    усреднением по каналам, чтобы сохранить пользу предобучения.
    '''
    def __init__(self, n_classes=3, freeze_backbone=True):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Шаги инициализации:
        # 1) перевести conv1 на 1‑канальный вход (усреднить RGB‑веса)
        # 2) заменить m.fc на линейный слой под n_classes
        # 3) (опц.) заморозить backbone, обучать только fc
        # заменить conv1 на 1-канальный, корректно инициализировать
        w = m.conv1.weight.data
        # Подменяем слой на 1 входной канал (рентгены — монохромные)
        # Адаптируем под один канал (рентген) вместо RGB
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            m.conv1.weight.copy_(w.mean(dim=1, keepdim=True))
        # заменить голову
        # Подменяем финальный классификатор под количество классов задачи
        # Меняем головной слой на число классов задачи
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        # заморозка
        if freeze_backbone:
            for name, p in m.named_parameters():
                p.requires_grad = ('fc' in name)
        self.net = m

    def forward(self, x):
        return self.net(x)

# Архитектура transfer-learning
class ResNet50Transfer(nn.Module):
    '''
    ResNet50 с transfer learning для 1-канальных изображений.

    Первый сверточный слой адаптируется к одному каналу, веса берутся как
    среднее по предобученным RGB-фильтрам; по умолчанию обучаем только fc.
    '''
    def __init__(self, n_classes=3, freeze_backbone=True):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Шаги инициализации:
        # 1) conv1 -> 1‑канальный вход (усреднить RGB‑веса)
        # 2) заменить fc под n_classes
        # 3) (опц.) заморозить backbone, обучать только fc
        # 1-канальный conv1 с корректной инициализацией средним по RGB-каналам
        w = m.conv1.weight.data
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            m.conv1.weight.copy_(w.mean(dim=1, keepdim=True))
        # замена 'головы'
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        # заморозка feature extractor (как в baseline: оставляем обучаемым лишь fc)
        if freeze_backbone:
            for name, p in m.named_parameters():
                p.requires_grad = ('fc' in name)
        self.net = m

    # методы разморозки слоев
    def unfreeze_layer4(self):
        '''Размораживает блок layer4 и fc для тонкой настройки верхних слоев.'''
        for name, p in self.net.named_parameters():
            if 'layer4' in name or 'fc' in name:
                p.requires_grad = True

    def unfreeze_layer3_4(self):
        '''Размораживает блоки layer3 и layer4 (и fc) для более глубокой адаптации.'''
        for name, p in self.net.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                p.requires_grad = True

    def forward(self, x):
        return self.net(x)

class ResNet50WithMeta(nn.Module):
    '''
    Модель, объединяющая признаки ResNet50 с табличными метаданными.

    Backbone выдает эмбеддинг картинки, простой MLP — эмбеддинг метаданных.
    Конкатенация эмбеддингов поступает в финальный классификатор.
    '''
    def __init__(self, n_classes=3, meta_dim=None, freeze_backbone=True):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Шаги инициализации:
        # 1) conv1 -> 1‑канальный вход (усреднить RGB‑веса)
        # 2) убрать финальный fc, оставить эмбеддинг (2048)
        # 3) (опц.) заморозить backbone
        # 4) MLP для мета‑признаков
        # 5) классификатор над конкатенацией [f_img, f_meta]
        # 1-канальный вход
        w = m.conv1.weight.data
        m.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        with torch.no_grad(): m.conv1.weight.copy_(w.mean(dim=1, keepdim=True))

        feat_dim = m.fc.in_features     # 2048
        m.fc = nn.Identity()            # возьмем фичи перед головой

        if freeze_backbone:
            for name,p in m.named_parameters():
                p.requires_grad = False

        self.backbone = m

        # простая MLP-голова для мета
        hid = 64
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, hid),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(feat_dim + hid, n_classes)

    def unfreeze_layer4(self):
        '''Размораживает последний блок backbone для тонкой настройки.'''
        for name,p in self.backbone.named_parameters():
            if 'layer4' in name: p.requires_grad = True

    def unfreeze_layer3_4(self):
        '''Размораживает блоки layer3 и layer4 для более глубокой адаптации.'''
        for name,p in self.backbone.named_parameters():
            if 'layer3' in name or 'layer4' in name: p.requires_grad = True

    # Шаги forward:
    # 1) извлечь f_img из backbone; 2) извлечь f_meta из MLP; 3) склеить и классифицировать
    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)           # [B,2048]
        f_meta = self.meta(x_meta)             # [B,64]
        f = torch.cat([f_img, f_meta], dim=1)  # [B,2112]
        return self.classifier(f)
    
def compute_class_weights(df, num_classes=3, device=None):
    '''
    Считает веса классов по обратной частоте для компенсации дисбаланса.

    Нормализует веса к сумме 1 и переносит на `device`.

    Args:
        df (pd.DataFrame): Таблица с колонкой `class_id`.
        num_classes (int): Число классов.
        device (torch.device|None): Целевое устройство для весов.

    Returns:
        torch.Tensor: Тензор весов классов длины `num_classes`.
    '''
    # Шаг 1. Подсчет частот классов и вычисление обратных частот как весов
    counts = Counter(df['class_id'])
    total = sum(counts.values())
    weights = torch.tensor([total / counts[i] for i in range(num_classes)], dtype=torch.float32)
    weights = (weights / weights.sum()).to(device)  # нормализуем веса
    print('Class distribution (train):', counts)
    print('Class weights:', weights.detach().cpu().numpy())
    return weights

# Запуск одной эпохи модели RestNet18 (цикла обучения)
def model_run_epoch(model_, scaler_, optimizer_, criterion_, loader, device_, is_train_=True, amp_enabled_=True):
    # Шаги: режим (train/val), контекст градиентов, forward с autocast, (опц.) backward через GradScaler, сбор метрик
    '''
    Проход по эпохе для модели без метаданных.

    Использует AMP (autocаст и GradScaler) на CUDA для ускорения и экономии памяти.
    Валидация выполняется без градиентов.
    '''
    model_.train(is_train_) # режим обучения
    total, correct, loss_sum = 0, 0, 0.0
    y_true_all, y_pred_all = [], []

    ctx_outer = torch.enable_grad() if is_train_ else torch.no_grad()
    with ctx_outer:
        for x, y in tqdm(loader, total=len(loader), desc=('train' if is_train_ else 'val'), leave=False):
            x, y = x.to(device_, non_blocking=True), torch.as_tensor(y, device=device_)
            with torch.amp.autocast('cuda', enabled=amp_enabled_):
                logits = model_(x)
                loss = criterion_(logits, y)

            # Обучение
            if is_train_:
                optimizer_.zero_grad(set_to_none=True) # обнуляем старые градиенты
                scaler_.scale(loss).backward()         # обратное распространение ошибки (градиенты)
                scaler_.step(optimizer_)               # обновляем веса
                scaler_.update()                       # обновляем масштаб для AMP

            preds = logits.argmax(1)
            bs = y.size(0)
            total += bs
            correct += (preds == y).sum().item()
            loss_sum += loss.item() * bs
            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return (loss_sum / total), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

def model_run_epoch_meta(model_, scaler_, optimizer_, criterion_, loader, device_, is_train_, amp_enabled_):
    # Шаги: режим (train/val), перенос (x_img, x_meta, y), forward с autocast, (опц.) backward через GradScaler, сбор метрик
    '''
    Проход по эпохе для модели, принимающей (x_img, x_meta).

    Повторяет логику model_run_epoch, но работает с картинкой и метаданными.
    '''
    if is_train_: model_.train()
    else:         model_.eval()

    tot_loss, y_true, y_pred = 0.0, [], []

    pbar = tqdm(loader, total=len(loader), leave=False, desc='train' if is_train_ else 'valid')
    for batch in pbar:
        x_img, x_meta, y = batch
        x_img  = x_img.to(device_, non_blocking=True)
        x_meta = x_meta.to(device_, non_blocking=True)
        y      = y.to(device_, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=amp_enabled_):
            logits = model_(x_img, x_meta)
            loss = criterion_(logits, y)

        if is_train_:
            optimizer_.zero_grad(set_to_none=True)
            scaler_.scale(loss).backward()
            scaler_.step(optimizer_)
            scaler_.update()

        tot_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        y_pred.append(preds.detach().cpu())
        y_true.append(y.detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    loss = tot_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro')
    return loss, acc, f1

# Обучение модели
def model_fit(model_, scaler_, optimizer_, criterion_, dl_tr_, dl_va_, device_, epochs_, amp_enabled_, ckpt_path_):
    '''
    Обучает модель N эпох и сохраняет лучший по F1 чекпоинт.

    На каждой эпохе считает train/val метрики и при улучшении F1 вал сохраняет веса.
    '''
    # Убедимся, что директория существует
    # Создаем каталог для чекпоинта, если не существует
    # Шаг 0. Готовим директорию для чекпоинта (если её нет)
    ckpt_path_.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    for e in range(1, epochs_+1):
        print(f'Epoch {e}/{epochs_}', flush=True)
        tr_loss, tr_acc, tr_f1 = model_run_epoch(model_=model_,
                                                     scaler_=scaler_,
                                                     optimizer_=optimizer_,
                                                     criterion_=criterion_,
                                                     loader=dl_tr_,
                                                     device_=device_,
                                                     is_train_=True,
                                                     amp_enabled_=amp_enabled_)
        va_loss, va_acc, va_f1 = model_run_epoch(model_=model_,
                                                     scaler_=scaler_,
                                                     optimizer_=optimizer_,
                                                     criterion_=criterion_,
                                                     loader=dl_va_,
                                                     device_=device_,
                                                     is_train_=False,
                                                     amp_enabled_=amp_enabled_)
        print(f'Epoch {e}/{epochs_} | train: loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | '
              f'val: loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}')
        # Шаг 2. Сохраняем лучший чекпоинт по валид. F1
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model_.state_dict(), ckpt_path_)
            print(f'[V] Сохранена лучшая модель в {ckpt_path_} (val F1={best_f1:.3f})')
    return best_f1

def model_fit_meta(model_, scaler_, optimizer_, criterion_, dl_tr_, dl_va_, device_, epochs_, ckpt_path_, amp_enabled_):
    '''
    Обучает модель с метаданными и сохраняет лучший по F1 чекпоинт.

    Использует `model_run_epoch_meta` и батчи (x_img, x_meta, y).
    '''
    best_f1 = -1.0
    for e in range(1, epochs_+1):
        print(f'Epoch {e}/{epochs_}', flush=True)
        # Шаг 1. Прогон train/val для модели с метаданными
        tr_loss, tr_acc, tr_f1 = model_run_epoch_meta(model_, scaler_, optimizer_, criterion_, dl_tr_, device_, True,  amp_enabled_)
        va_loss, va_acc, va_f1 = model_run_epoch_meta(model_, scaler_, None,       criterion_, dl_va_, device_, False, amp_enabled_)
        print(f'Epoch {e}/{epochs_} | train: loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | '
              f'val: loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}')
        # Шаг 2. Сохраняем лучший чекпоинт по валид. F1
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model_.state_dict(), ckpt_path_)
            print(f'[V] Сохранена лучшая модель в {ckpt_path_} (val F1={best_f1:.3f})')
    return best_f1

def predictions_on_validation(model_, device_, dl_va_):
    '''
    Инференс на валидации: возвращает (y_true, y_pred, y_prob).

    Считает softmax-вероятности и предсказанные классы для батчей.
    '''
    # Собираем предсказания на валидации
    # Шаг 1. Сбор предсказаний на валидации (без метаданных)
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device_.type=='cuda')):
        for x, y in tqdm(dl_va_, total=len(dl_va_), desc='Validation inference', leave=False):
            # Шаг 2. Прямой проход и сохранение softmax/argmax
            x = x.to(device_, non_blocking=True)
            logits = model_(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred  = probs.argmax(1)

            y_prob.append(probs)
            y_pred.append(pred)
            y_true.append(y.numpy())

    # Шаг 3. Склеиваем батчи в итоговые массивы
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    return y_true, y_pred, y_prob

def predictions_on_validation_meta(model_, device_, dl_va_):
    '''
    Инференс на валидации для модели с метаданными.

    Принимает батчи (x_img, x_meta, y); возвращает массивы истинных,
    предсказанных классов и вероятностей softmax.
    '''
    # Шаг 1. Сбор предсказаний на валидации (с метаданными)
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device_.type=='cuda')):
        for x_img, x_meta, y in tqdm(dl_va_, total=len(dl_va_), desc='Validation inference', leave=False):
            # Шаг 2. Прямой проход и сохранение softmax/argmax
            x_img  = x_img.to(device_, non_blocking=True)
            x_meta = x_meta.to(device_, non_blocking=True)
            logits = model_(x_img, x_meta)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            pred   = probs.argmax(1)
            y_prob.append(probs)
            y_pred.append(pred)
            y_true.append(y.numpy())

    # Шаг 3. Склеиваем батчи в итоговые массивы
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    return y_true, y_pred, y_prob
