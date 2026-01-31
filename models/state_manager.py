import threading


class StateManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Ініціалізація всіх полів стану."""
        self.model = {}
        self.df = None
        self.data = None
        self.L = 0
        self.V = 0
        self.new_ngram = None
        self.uploaded_files = {}
        self.file_lengths = {}
        self.batch_results = []
        
        # UI state
        self.toast_visible = False
        self.error_visible = False
        self.analyze_visible = False
        self.length_updated = False
        self.clicked_ngram = None
        self.save_folder = None

    def reset(self):
        """Скидає стан до початкового."""
        self._initialize()

    def clear_model(self):
        """Очищує модель."""
        if isinstance(self.model, dict):
            self.model.clear()
        self.model = {}

    def clear_data(self):
        """Очищує дані."""
        self.data = None
        self.df = None
        self.new_ngram = None

state = StateManager()