"""
Конфігураційні константи для програми.
"""

# ============================================================
# WINDOW PARAMETERS (DEFAULT)
# ============================================================
DEFAULT_WINDOW_MIN = 10
DEFAULT_WINDOW_STEP = 1
DEFAULT_WINDOW_EXPANSION = 1
DEFAULT_WINDOW_MAX = 100

# ============================================================
# ANALYSIS PARAMETERS
# ============================================================
DEFAULT_N_SIZE = 1  # Порядок n-грами (1 = уніграма)
DEFAULT_F_MIN = 3   # Мінімальна частота для включення в аналіз
DEFAULT_MIN_DIST = 1  # Мінімальна відстань (0 або 1)
DEFAULT_POLYNOM_DEGREE = 1  # Степінь полінома для DFA

# ============================================================
# BOUNDARY CONDITIONS
# ============================================================
BOUNDARY_NO = "no"
BOUNDARY_ORDINARY = "ordinary"
BOUNDARY_PERIODIC = "periodic"
DEFAULT_BOUNDARY = BOUNDARY_ORDINARY

# ============================================================
# OVERLAP MODES
# ============================================================
OVERLAP_OVERLAPPING = "overlapping"
OVERLAP_NON_OVERLAPPING = "non-overlapping"
DEFAULT_OVERLAP_MODE = OVERLAP_OVERLAPPING

# ============================================================
# SPLIT MODES
# ============================================================
SPLIT_WORD = "word"
SPLIT_SYMBOL = "symbol"
SPLIT_LETTER = "letter"
DEFAULT_SPLIT_MODE = SPLIT_WORD

# ============================================================
# ALGORITHM MODES
# ============================================================
ALGO_FA = "1"   # Fluctuation Analysis
ALGO_DFA = "2"  # Detrended Fluctuation Analysis
DEFAULT_ALGO = ALGO_FA

# ============================================================
# DEFINITION MODES
# ============================================================
DEFINITION_STATIC = "static"
DEFINITION_DYNAMIC = "dynamic"
DEFAULT_DEFINITION = DEFINITION_STATIC

# ============================================================
# SCALE MODES
# ============================================================
SCALE_LINEAR = "linear"
SCALE_LOG = "log"
DEFAULT_SCALE = SCALE_LOG

# ============================================================
# UI COLORS
# ============================================================
COLORS = {
    "background": "#a1a1a1",
    "text": "#a1a1a1",
    "primary": "#007bff",
    "success": "#28a745",
    "danger": "#dc3545",
    "warning": "#ffc107",
    "info": "#17a2b8"
}

# ============================================================
# FILE PROCESSING
# ============================================================
SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.md', '.rst']
SUPPORTED_CODE_EXTENSIONS = [
    '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt',
    '.html', '.css', '.scss', '.sass', '.less',
    '.json', '.xml', '.yaml', '.yml', '.toml',
    '.sql', '.sh', '.bash', '.ps1', '.bat'
]

# ============================================================
# BATCH PROCESSING
# ============================================================
DEFAULT_FMIN1 = 3
DEFAULT_FMIN2 = 10
BATCH_WINDOW_MODE_MANUAL = "manual"
BATCH_WINDOW_MODE_AUTO = "auto"
DEFAULT_BATCH_WINDOW_MODE = BATCH_WINDOW_MODE_AUTO

# ============================================================
# APP SETTINGS
# ============================================================
APP_TITLE = "FA/DFA Analysis Tool"
APP_DEBUG = False
APP_PORT = 8050
APP_HOST = "0.0.0.0"