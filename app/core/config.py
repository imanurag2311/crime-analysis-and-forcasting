"""
Centralized Configuration for Mumbai Crime Analysis System
===========================================================
All project-wide constants, paths, and tunable parameters are defined here.
Do not scatter hard-coded values across modules.
"""

from pathlib import Path
import os

# ============================================================
# PROJECT ROOT & DIRECTORY PATHS
# ============================================================

# Project root is two levels up from this config file (app/core/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UPLOADS_DIR = DATA_DIR / "uploads"

# Database
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "crime_database.db"

# Models
MODEL_DIR = PROJECT_ROOT / "models"
SARIMA_MODEL_DIR = MODEL_DIR / "sarima"

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OCR_OUTPUT_DIR = OUTPUT_DIR / "ocr"
FORECAST_OUTPUT_DIR = OUTPUT_DIR / "forecasts"
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"
LOG_DIR = OUTPUT_DIR / "logs"

# ============================================================
# ENSURE DIRECTORIES EXIST
# ============================================================

ALL_DIRS = [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, UPLOADS_DIR,
    DATABASE_DIR, SARIMA_MODEL_DIR,
    OCR_OUTPUT_DIR, FORECAST_OUTPUT_DIR, METRICS_OUTPUT_DIR, LOG_DIR
]

for d in ALL_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# OPENCV / IMAGE PREPROCESSING
# ============================================================

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Sharpening kernel
SHARPENING_KERNEL = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]

# Gaussian blur kernel size
GAUSSIAN_BLUR_KERNEL = (3, 3)

# Deskew angle limits (degrees)
DESKEW_MIN_ANGLE = -15.0
DESKEW_MAX_ANGLE = 15.0

# PDF rendering DPI
PDF_RENDER_DPI = 300

# ============================================================
# OCR SETTINGS
# ============================================================

OCR_ROW_Y_TOLERANCE = 12.0
OCR_CONFIDENCE_THRESHOLD = 0.3  # Below this = flagged
OCR_LANGUAGES = ['en']

# ============================================================
# SARIMA CONFIGURATION
# ============================================================

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 0, 12)
SEASONAL_PERIOD = 12

# Train/test split ratio
TRAIN_TEST_RATIO = 0.80

# Forecast horizon
DEFAULT_FORECAST_HORIZON = 6
MIN_FORECAST_HORIZON = 3
MAX_FORECAST_HORIZON = 12

# Minimum observations needed for SARIMA
MIN_FORECAST_HISTORY = 24  # At least 24 months (2 full seasonal cycles)

# Confidence intervals
CONFIDENCE_LEVELS = [0.80, 0.95]

# ============================================================
# DATABASE SETTINGS
# ============================================================

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# ============================================================
# DATA SCHEMA - 10 PRIMARY PROJECT COLUMNS
# ============================================================

PRIMARY_COLUMNS = [
    "crime_category",
    "month",
    "year",
    "case_count",
    "legal_section",
    "page",
    "source_file",
    "standard_group",
    "domain",
    "legal_section_bns"
]

# Internal metadata columns
METADATA_COLUMNS = [
    "record_id",
    "created_at",
    "updated_at",
    "data_source",
    "validation_status",
    "ocr_confidence"
]

# ============================================================
# CRIME DOMAINS
# ============================================================

CRIME_DOMAINS = [
    "General IPC",
    "Crimes Against Women (CAW)",
    "Economic Offences (EOW)",
    "NDPS Act",
    "Cyber Crime",
    "Property Crime",
    "Violent Crime",
    "Other"
]

# ============================================================
# FILE UPLOAD SETTINGS
# ============================================================

ALLOWED_UPLOAD_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.csv'}
MAX_UPLOAD_SIZE_MB = 50

# ============================================================
# ROLLING AVERAGE WINDOWS
# ============================================================

ROLLING_WINDOW_3 = 3
ROLLING_WINDOW_6 = 6

# ============================================================
# LOGGING
# ============================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOG_DIR / "application.log"

# ============================================================
# DASHBOARD SETTINGS
# ============================================================

DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "Crime Analysis & Forecasting System"

# ============================================================
# DATE EXTRACTION PATTERNS
# ============================================================

# Regex patterns for extracting dates from PDF headers
DATE_PATTERNS = [
    r'(\d{2})[\/\.](\d{2})[\/\.](\d{4})',  # DD/MM/YYYY or DD.MM.YYYY
    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',
]

MONTH_NAME_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
    'oct': 10, 'nov': 11, 'dec': 12
}

# ============================================================
# OCR FILTERING KEYWORDS
# ============================================================

SUMMARY_ROW_KEYWORDS = [
    'total', 'grand total', 'sub total', 'subtotal',
    'overall', 'aggregate', 'sum total', 'all total'
]

MEASUREMENT_KEYWORDS = [
    'kg', 'kilogram', 'litre', 'liter', 'gram', 'gm',
    'ml', 'millilitre', 'ton', 'tonne', 'quintal'
]

SUB_BREAKDOWN_KEYWORDS = [
    'pocso', 'pita', 'sub-clause', 'sub clause',
    'section', 'clause', 'sub-section'
]

# ============================================================
# SYNTHETIC DATA LABEL
# ============================================================

SYNTHETIC_DATA_LABEL = "SYNTHETIC TEST DATA"
