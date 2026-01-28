import gc
import pandas as pd
from typing import List
from models.state_manager import state_manager

def clear_memory(keep: List[str] = []):
    heavy_attributes = {
        'model': {},          # Словник з об'єктами Ngram
        'df': None,           # Pandas DataFrame з результатами
        'data': [],           # Список токенів (слова/символи)
        'new_ngram': None,    # Об'єкт newNgram для динамічного режиму
        'batch_results': []   # Список результатів пакетної обробки
    }

    for attr, default_value in heavy_attributes.items():
        if attr not in keep:
            current_val = getattr(state_manager, attr, None)
            
            if isinstance(current_val, dict):
                current_val.clear()
            elif isinstance(current_val, list):
                current_val.clear()
            
            setattr(state_manager, attr, default_value)

    try:
        from data.text_processing import prepare_data
        if hasattr(prepare_data, 'clear_cache'):
            prepare_data.clear_cache()
            
        from core.legacy_indexer import build_static_index
        if hasattr(build_static_index, 'clear_cache'):
            build_static_index.clear_cache()
    except ImportError:
        pass

    gc.collect(generation=2)
    gc.collect(generation=1)
    gc.collect(generation=0)

def get_memory_usage():
    import sys
    usage = {}
    if state_manager.df is not None:
        usage['df_memory'] = state_manager.df.memory_usage(deep=True).sum() / (1024**2)
    usage['model_size'] = sys.getsizeof(state_manager.model) / (1024**2)
    usage['data_len'] = len(state_manager.data)
    return usage