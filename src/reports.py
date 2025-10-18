import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score,
    recall_score, precision_score,
    average_precision_score, roc_auc_score,
    classification_report,
)

def save_model_report(model_name, y_true, y_pred, y_prob, target_names, num_epocs=0, csv_path=''):
    '''
    Сохранение и накопление сводных метрик модели в CSV.

    Args:
        model_name (str): название модели/эксперимента.
        y_true (array-like): истинные метки классов (shape: N, значения 0..C-1).
        y_pred (array-like): предсказанные метки классов (shape: N).
        y_prob (ndarray|None): вероятности классов (shape: N x C), если нужно посчитать AUC/PR/Brier.
        target_names (list[str]): имена классов (для построчных метрик report).
        num_epocs (int): количество эпох обучения (для протокола).
        csv_path (pathlib.Path|str): путь к CSV, куда добавляем строку с метриками.
    '''

    # Шаг 1. Базовые агрегированные метрики по всему набору
    summary = {
        'model': f'{model_name}',
        'epocs': int(num_epocs),
        'timestamp': dt.datetime.now().isoformat(timespec='seconds'),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'f1_micro': float(f1_score(y_true, y_pred, average='micro')),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
        'recall_micro': float(recall_score(y_true, y_pred, average='micro')),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
        'precision_micro': float(precision_score(y_true, y_pred, average='micro')),
    }

    # Шаг 2. Построчные метрики по классам из classification_report
    # (добавляем f1/recall/precision/support для каждого класса)
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    for name in target_names:
        if name in report_dict:
            if 'f1-score' in report_dict[name]:
                summary[f'f1_{name}'] = float(report_dict[name]['f1-score'])
            if 'recall' in report_dict[name]:
                summary[f'recall_{name}'] = float(report_dict[name]['recall'])
            if 'precision' in report_dict[name]:
                summary[f'precision_{name}'] = float(report_dict[name]['precision'])
            if 'support' in report_dict[name]:
                summary[f'support_{name}'] = int(report_dict[name]['support'])

    # Шаг 3. ROC-AUC / PR-AUC / Brier score (если заданы вероятности классов)
    if 'y_prob' in locals() and y_prob is not None:
        try:
            # 3.1. Готовим one-hot для вычисления многоклассового AUC/PR
            n_classes = y_prob.shape[1]
            y_true_oh = np.eye(n_classes)[y_true]
            # 3.2. ROC-AUC (OvR и OvO) и PR-AUC (macro)
            summary['roc_auc_ovr_macro'] = float(
                roc_auc_score(y_true_oh, y_prob, average='macro', multi_class='ovr')
            )
            summary['roc_auc_ovo_macro'] = float(
                roc_auc_score(y_true_oh, y_prob, average='macro', multi_class='ovo')
            )
            summary['pr_auc_macro'] = float(
                average_precision_score(y_true_oh, y_prob, average='macro')
            )
            # 3.3. (Необязательная) метрика калибровки — Brier score
            summary['brier'] = float(np.mean(np.sum((y_true_oh - y_prob) ** 2, axis=1)))
        except Exception as e:
            print('[WARN] AUC/PR-AUC calc skipped:', e)

    # Шаг 4. Превращаем словарь метрик в DataFrame (одна строка)
    df_metrics = pd.DataFrame([summary])

    # Шаг 5. Накапливаем метрики в CSV (добавляем строку к существующим)
    csv_path = Path(csv_path)
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        prev = pd.concat([prev, df_metrics], ignore_index=True)
        prev.to_csv(csv_path, index=False)
    else:
        df_metrics.to_csv(csv_path, index=False)

    # Шаг 6. Короткий лог о том, куда записали метрики
    print('Сводные метрики сохранены в CSV:', csv_path)