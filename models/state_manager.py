import pandas as pd
from typing import Dict, Any, List, Optional
from models.ngram import Ngram, newNgram

class StateManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self.uploaded_files: Dict[str, str] = {}
        self.file_lengths: Dict[str, Dict[str, int]] = {}
        self.batch_results: List[Dict[str, Any]] = []
        self.model: Dict[Any, Ngram] = {}
        self.df: Optional[pd.DataFrame] = None
        self.data: List[Any] = []
        self.new_ngram: Optional[newNgram] = None
        self.save_folder: Optional[str] = None
        self.L: int = 0
        self.V: int = 0

    def reset_analysis_state(self):
        self.model = {}
        self.df = None
        self.new_ngram = None
        self.data = []

state_manager = StateManager()