import gc
from typing import List


def clear_memory(keep: List[str] = None):
    if keep is None:
        keep = []

    try:
        from data.text_processing import prepare_data
        if hasattr(prepare_data, 'clear_cache') and 'prepare_data_cache' not in keep:
            prepare_data.clear_cache()
    except Exception:
        pass

    try:
        from core.legacy_indexer import make_markov_chain
        if hasattr(make_markov_chain, 'clear_cache') and 'make_markov_chain_cache' not in keep:
            make_markov_chain.clear_cache()
    except Exception:
        pass

    gc.collect(generation=2)
    gc.collect(generation=1)
    gc.collect(generation=0)
