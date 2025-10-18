import torch                       # основной фреймворк PyTorch для нейронных сетей
from torchvision import transforms  # стандартные трансформации изображений (resize, flip, tensor)
from torch.utils.data import DataLoader  # классы для создания датасетов и загрузчиков данных
from sklearn.metrics import f1_score
from src.models import RSNADataset

def dataloaders_for_optuna(img_size, bs_tr, bs_va, df_train, df_val, img_dir):
    '''
    Создает train/val DataLoader с трансформациями для экспериментов Optuna.

    Args:
        img_size (int): Размер стороны квадратного изображения.
        bs_tr (int): Batch size для обучения.
        bs_va (int): Batch size для валидации.
        df_train (pd.DataFrame): Разметка train (patientId, class_id).
        df_val (pd.DataFrame): Разметка val.
        img_dir (Path|str): Папка с PNG-снимками.

    Returns:
        tuple[DataLoader, DataLoader]: Пара (dl_train, dl_val).
    '''
    # Тренировочные аугментации для улучшения обобщения
    # Шаг 1. Трансформации для train (лёгкие аугментации)
    tfm_train_local = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor()
    ])
    # Валидационные (тестовые) аугментации для улучшения обобщения
    # Шаг 2. Трансформации для val (без аугментаций)
    tfm_val_local = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    # Создаем датасеты и соответствующие загрузчики
    # Шаг 3. Датасеты с указанными трансформациями
    ds_tr = RSNADataset(df_train, img_dir, tfm_train_local)
    ds_va = RSNADataset(df_val, img_dir, tfm_val_local)
    # pin_memory ускоряет перенос батчей на GPU; shuffle только для train
    # Шаг 4. DataLoader'ы (pin_memory — быстрее копирование на GPU; shuffle — только для train)
    dl_tr_local = DataLoader(ds_tr, batch_size=bs_tr, shuffle=True,  num_workers=0, pin_memory=True)
    dl_va_local = DataLoader(ds_va, batch_size=bs_va, shuffle=False, num_workers=0, pin_memory=True)
    return dl_tr_local, dl_va_local

def resnet50_run_epoch_optuna(model, loader, criterion, scaler, device, optimizer=None, amp_flag=True):
    '''
    Одна эпоха train/val с поддержкой AMP на CUDA.

    Если optimizer задан — train, иначе val. Возвращает (avg_loss, macro F1).
    '''
    # Выбираем режим: обучение или валидация
    is_train = optimizer is not None
    model.train(is_train)  # управляет Dropout/BatchNorm
    y_true, y_pred = [], []
    total, loss_sum = 0, 0.0
    # На валидации выключаем градиенты для скорости и экономии памяти
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            # Переносим батч на целевое устройство
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            # Autocast ускоряет вычисления на CUDA и экономит память
            with torch.amp.autocast('cuda', enabled=amp_flag):
                logits = model(x)
                loss = criterion(logits, y)
            if is_train:
                # Шаг оптимизации с GradScaler для AMP-стабильности
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # 3.4. Предсказания и накопление статистик
            preds = logits.argmax(1)  # предсказанный класс по максимуму логита
            bs = y.size(0)
            total += bs; loss_sum += loss.item() * bs
            y_true += y.detach().cpu().tolist()
            y_pred += preds.detach().cpu().tolist()
    # Итоговая метрика: macro F1 по всем классам
    # Шаг 4. Итоговая метрика за эпоху
    f1 = f1_score(y_true, y_pred, average='macro') if total > 0 else 0.0
    return (loss_sum / max(1, total)), f1
