"""
SQLite Database Layer
======================
Central data store for the Mumbai Crime Analysis System.
Provides CRUD operations, dataset versioning, and audit logging.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

from app.core.config import DATABASE_PATH, DATABASE_URL, PRIMARY_COLUMNS

logger = logging.getLogger(__name__)


class CrimeDatabase:
    """
    Central SQLite database manager.
    
    Handles all database operations including CRUD for crime records,
    forecast results, model metrics, processing logs, and dataset versions.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path or DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self._initialize_tables()
        logger.info(f"Database initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a raw sqlite3 connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_tables(self):
        """Create all required tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Crime records - main data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crime_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                crime_category TEXT NOT NULL,
                month INTEGER NOT NULL CHECK(month >= 1 AND month <= 12),
                year INTEGER NOT NULL CHECK(year >= 1900 AND year <= 2100),
                case_count INTEGER NOT NULL CHECK(case_count >= 0),
                legal_section TEXT,
                page INTEGER,
                source_file TEXT,
                standard_group TEXT,
                domain TEXT,
                legal_section_bns TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'manual',
                validation_status TEXT DEFAULT 'valid',
                ocr_confidence REAL,
                dataset_version INTEGER DEFAULT 1,
                UNIQUE(crime_category, month, year, legal_section)
            )
        """)

        # Forecast results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_results (
                forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
                crime_category TEXT NOT NULL,
                forecast_date TEXT NOT NULL,
                forecast_value REAL NOT NULL,
                lower_80 REAL,
                upper_80 REAL,
                lower_95 REAL,
                upper_95 REAL,
                model_version TEXT,
                dataset_version INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Model metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                crime_category TEXT NOT NULL,
                rmse REAL,
                mae REAL,
                mape REAL,
                r_squared REAL,
                sarima_order TEXT,
                sarima_seasonal_order TEXT,
                training_start TEXT,
                training_end TEXT,
                train_size INTEGER,
                test_size INTEGER,
                dataset_version INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Processing logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                page_number INTEGER,
                rows_detected INTEGER DEFAULT 0,
                rows_accepted INTEGER DEFAULT 0,
                rows_rejected INTEGER DEFAULT 0,
                rejection_reasons TEXT,
                processing_type TEXT,
                status TEXT DEFAULT 'completed',
                error_message TEXT,
                processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Dataset versions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_number INTEGER NOT NULL,
                record_count INTEGER NOT NULL DEFAULT 0,
                update_type TEXT NOT NULL,
                action_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # OCR records (optional, for audit)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_records (
                ocr_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                page_number INTEGER,
                raw_text TEXT,
                parsed_category TEXT,
                parsed_count INTEGER,
                confidence REAL,
                status TEXT DEFAULT 'pending',
                rejection_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Database tables initialized")

    # ================================================================
    # CRIME RECORDS CRUD
    # ================================================================

    def insert_records(self, df: pd.DataFrame, data_source: str = "manual",
                       dataset_version: Optional[int] = None) -> Dict[str, int]:
        """
        Insert multiple crime records from a DataFrame.
        
        Returns dict with counts: inserted, duplicates, errors
        """
        if dataset_version is None:
            dataset_version = self.get_current_version()

        inserted = 0
        duplicates = 0
        errors = 0

        conn = self._get_connection()
        cursor = conn.cursor()

        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO crime_records 
                    (crime_category, month, year, case_count, legal_section,
                     page, source_file, standard_group, domain, legal_section_bns,
                     data_source, validation_status, ocr_confidence, dataset_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get('crime_category', ''),
                    int(row.get('month', 0)),
                    int(row.get('year', 0)),
                    int(row.get('case_count', 0)),
                    row.get('legal_section', ''),
                    row.get('page', None),
                    row.get('source_file', ''),
                    row.get('standard_group', ''),
                    row.get('domain', ''),
                    row.get('legal_section_bns', ''),
                    data_source,
                    row.get('validation_status', 'valid'),
                    row.get('ocr_confidence', None),
                    dataset_version
                ))
                inserted += 1
            except sqlite3.IntegrityError:
                duplicates += 1
                logger.warning(
                    f"Duplicate record: {row.get('crime_category')} "
                    f"{row.get('month')}/{row.get('year')}"
                )
            except Exception as e:
                errors += 1
                logger.error(f"Error inserting record: {e}")

        conn.commit()
        conn.close()

        result = {"inserted": inserted, "duplicates": duplicates, "errors": errors}
        logger.info(f"Insert results: {result}")
        return result

    def get_all_records(self) -> pd.DataFrame:
        """Get all crime records as a DataFrame."""
        conn = self._get_connection()
        df = pd.read_sql_query("SELECT * FROM crime_records ORDER BY year, month", conn)
        conn.close()
        return df

    def get_records_filtered(self, year: Optional[int] = None,
                              month: Optional[int] = None,
                              category: Optional[str] = None,
                              domain: Optional[str] = None) -> pd.DataFrame:
        """Get filtered crime records."""
        query = "SELECT * FROM crime_records WHERE 1=1"
        params = []

        if year is not None:
            query += " AND year = ?"
            params.append(year)
        if month is not None:
            query += " AND month = ?"
            params.append(month)
        if category is not None:
            query += " AND crime_category = ?"
            params.append(category)
        if domain is not None:
            query += " AND domain = ?"
            params.append(domain)

        query += " ORDER BY year, month, crime_category"

        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def update_record(self, record_id: int, updates: Dict[str, Any]) -> bool:
        """Update a specific crime record."""
        if not updates:
            return False

        set_clauses = []
        values = []
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)

        set_clauses.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(record_id)

        query = f"UPDATE crime_records SET {', '.join(set_clauses)} WHERE record_id = ?"

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, values)
            conn.commit()
            success = cursor.rowcount > 0
            if success:
                logger.info(f"Record {record_id} updated")
            return success
        except Exception as e:
            logger.error(f"Error updating record {record_id}: {e}")
            return False
        finally:
            conn.close()

    def delete_records(self, record_ids: List[int]) -> int:
        """Delete records by their IDs. Returns count of deleted records."""
        if not record_ids:
            return 0

        placeholders = ','.join(['?' for _ in record_ids])
        query = f"DELETE FROM crime_records WHERE record_id IN ({placeholders})"

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, record_ids)
            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"Deleted {deleted} records")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting records: {e}")
            return 0
        finally:
            conn.close()

    def delete_records_by_month(self, year: int, month: int,
                                 category: Optional[str] = None) -> int:
        """Delete records for a specific month/year, optionally filtered by category."""
        query = "DELETE FROM crime_records WHERE year = ? AND month = ?"
        params = [year, month]

        if category:
            query += " AND crime_category = ?"
            params.append(category)

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"Deleted {deleted} records for {month}/{year}")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting records: {e}")
            return 0
        finally:
            conn.close()

    def get_unique_categories(self) -> List[str]:
        """Get list of unique crime categories."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT crime_category FROM crime_records ORDER BY crime_category"
        )
        categories = [row[0] for row in cursor.fetchall()]
        conn.close()
        return categories

    def get_unique_years(self) -> List[int]:
        """Get list of unique years."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT year FROM crime_records ORDER BY year")
        years = [row[0] for row in cursor.fetchall()]
        conn.close()
        return years

    def get_unique_domains(self) -> List[str]:
        """Get list of unique domains."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT domain FROM crime_records WHERE domain IS NOT NULL ORDER BY domain")
        domains = [row[0] for row in cursor.fetchall()]
        conn.close()
        return domains

    def get_monthly_summary(self, category: Optional[str] = None) -> pd.DataFrame:
        """Get monthly aggregated case counts."""
        query = """
            SELECT year, month, crime_category, SUM(case_count) as total_cases
            FROM crime_records
            WHERE validation_status = 'valid'
        """
        params = []
        if category:
            query += " AND crime_category = ?"
            params.append(category)

        query += " GROUP BY year, month, crime_category ORDER BY year, month"

        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_record_count(self) -> int:
        """Get total number of records."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM crime_records")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_latest_month_year(self) -> Tuple[Optional[int], Optional[int]]:
        """Get the latest month and year in the dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT month, year FROM crime_records ORDER BY year DESC, month DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0], row[1]
        return None, None

    def check_duplicate(self, crime_category: str, month: int, year: int,
                        legal_section: Optional[str] = None) -> bool:
        """Check if a record already exists."""
        query = """
            SELECT COUNT(*) FROM crime_records 
            WHERE crime_category = ? AND month = ? AND year = ?
        """
        params = [crime_category, month, year]
        if legal_section:
            query += " AND legal_section = ?"
            params.append(legal_section)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0

    # ================================================================
    # FORECAST RESULTS
    # ================================================================

    def save_forecasts(self, forecasts: pd.DataFrame, crime_category: str,
                       dataset_version: Optional[int] = None):
        """Save forecast results to database."""
        if dataset_version is None:
            dataset_version = self.get_current_version()

        # Clear old forecasts for this category
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM forecast_results WHERE crime_category = ?",
            (crime_category,)
        )

        for _, row in forecasts.iterrows():
            cursor.execute("""
                INSERT INTO forecast_results 
                (crime_category, forecast_date, forecast_value,
                 lower_80, upper_80, lower_95, upper_95,
                 model_version, dataset_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                crime_category,
                str(row.get('forecast_date', '')),
                float(row.get('forecast_value', 0)),
                row.get('lower_80'),
                row.get('upper_80'),
                row.get('lower_95'),
                row.get('upper_95'),
                'SARIMA',
                dataset_version
            ))

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(forecasts)} forecasts for {crime_category}")

    def get_forecasts(self, crime_category: Optional[str] = None) -> pd.DataFrame:
        """Get forecast results."""
        query = "SELECT * FROM forecast_results"
        params = []
        if crime_category:
            query += " WHERE crime_category = ?"
            params.append(crime_category)
        query += " ORDER BY forecast_date"

        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    # ================================================================
    # MODEL METRICS
    # ================================================================

    def save_metrics(self, metrics: Dict[str, Any], crime_category: str,
                     dataset_version: Optional[int] = None):
        """Save model evaluation metrics."""
        if dataset_version is None:
            dataset_version = self.get_current_version()

        # Clear old metrics for this category
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM model_metrics WHERE crime_category = ?",
            (crime_category,)
        )

        cursor.execute("""
            INSERT INTO model_metrics 
            (crime_category, rmse, mae, mape, r_squared,
             sarima_order, sarima_seasonal_order,
             training_start, training_end, train_size, test_size,
             dataset_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            crime_category,
            metrics.get('rmse'),
            metrics.get('mae'),
            metrics.get('mape'),
            metrics.get('r_squared'),
            str(metrics.get('sarima_order', '')),
            str(metrics.get('sarima_seasonal_order', '')),
            str(metrics.get('training_start', '')),
            str(metrics.get('training_end', '')),
            metrics.get('train_size'),
            metrics.get('test_size'),
            dataset_version
        ))

        conn.commit()
        conn.close()
        logger.info(f"Saved metrics for {crime_category}")

    def get_metrics(self, crime_category: Optional[str] = None) -> pd.DataFrame:
        """Get model metrics."""
        query = "SELECT * FROM model_metrics"
        params = []
        if crime_category:
            query += " WHERE crime_category = ?"
            params.append(crime_category)

        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    # ================================================================
    # PROCESSING LOGS
    # ================================================================

    def log_processing(self, source_file: str, page_number: Optional[int],
                       rows_detected: int, rows_accepted: int, rows_rejected: int,
                       rejection_reasons: str = "", processing_type: str = "ocr",
                       status: str = "completed", error_message: str = ""):
        """Log a processing event."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO processing_logs 
            (source_file, page_number, rows_detected, rows_accepted, rows_rejected,
             rejection_reasons, processing_type, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            source_file, page_number, rows_detected, rows_accepted, rows_rejected,
            rejection_reasons, processing_type, status, error_message
        ))
        conn.commit()
        conn.close()

    def get_processing_logs(self) -> pd.DataFrame:
        """Get all processing logs."""
        conn = self._get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM processing_logs ORDER BY processing_timestamp DESC", conn
        )
        conn.close()
        return df

    # ================================================================
    # DATASET VERSIONING
    # ================================================================

    def get_current_version(self) -> int:
        """Get the current dataset version number."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(version_number) FROM dataset_versions"
        )
        result = cursor.fetchone()[0]
        conn.close()
        return result if result is not None else 0

    def create_version(self, update_type: str, action_summary: str = "") -> int:
        """Create a new dataset version and return version number."""
        current = self.get_current_version()
        new_version = current + 1
        record_count = self.get_record_count()

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dataset_versions 
            (version_number, record_count, update_type, action_summary)
            VALUES (?, ?, ?, ?)
        """, (new_version, record_count, update_type, action_summary))
        conn.commit()
        conn.close()

        logger.info(f"Created dataset version {new_version}: {update_type}")
        return new_version

    def get_version_history(self) -> pd.DataFrame:
        """Get dataset version history."""
        conn = self._get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM dataset_versions ORDER BY version_number DESC", conn
        )
        conn.close()
        return df

    # ================================================================
    # OCR RECORDS
    # ================================================================

    def save_ocr_records(self, records: List[Dict[str, Any]]):
        """Save OCR-extracted records for audit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        for rec in records:
            cursor.execute("""
                INSERT INTO ocr_records 
                (source_file, page_number, raw_text, parsed_category,
                 parsed_count, confidence, status, rejection_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec.get('source_file', ''),
                rec.get('page_number'),
                rec.get('raw_text', ''),
                rec.get('parsed_category', ''),
                rec.get('parsed_count'),
                rec.get('confidence'),
                rec.get('status', 'pending'),
                rec.get('rejection_reason', '')
            ))
        conn.commit()
        conn.close()

    def get_ocr_records(self, status: Optional[str] = None) -> pd.DataFrame:
        """Get OCR records, optionally filtered by status."""
        query = "SELECT * FROM ocr_records"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"

        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    # ================================================================
    # BACKUP / EXPORT
    # ================================================================

    def export_to_excel(self, filepath: Path):
        """Export all crime records to Excel."""
        df = self.get_all_records()
        # Keep only primary + useful metadata columns
        export_cols = [c for c in PRIMARY_COLUMNS if c in df.columns]
        export_cols += ['record_id', 'created_at', 'updated_at', 'data_source']
        export_cols = [c for c in export_cols if c in df.columns]
        df[export_cols].to_excel(filepath, index=False, engine='openpyxl')
        logger.info(f"Exported {len(df)} records to {filepath}")

    def export_to_csv(self, filepath: Path):
        """Export all crime records to CSV."""
        df = self.get_all_records()
        export_cols = [c for c in PRIMARY_COLUMNS if c in df.columns]
        export_cols += ['record_id', 'created_at', 'updated_at', 'data_source']
        export_cols = [c for c in export_cols if c in df.columns]
        df[export_cols].to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} records to {filepath}")

    def backup_database(self, backup_path: Path):
        """Create a backup of the SQLite database."""
        import shutil
        shutil.copy2(str(self.db_path), str(backup_path))
        logger.info(f"Database backed up to {backup_path}")

    # ================================================================
    # STATISTICS
    # ================================================================

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for dashboard display."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM crime_records")
        total_records = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT crime_category) FROM crime_records")
        total_categories = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(year), MAX(year) FROM crime_records")
        row = cursor.fetchone()
        min_year = row[0]
        max_year = row[1]

        cursor.execute(
            "SELECT month, year FROM crime_records ORDER BY year DESC, month DESC LIMIT 1"
        )
        latest = cursor.fetchone()
        latest_month = latest[0] if latest else None
        latest_year = latest[1] if latest else None

        cursor.execute("SELECT COUNT(*) FROM model_metrics")
        trained_models = cursor.fetchone()[0]

        cursor.execute(
            "SELECT MAX(created_at) FROM model_metrics"
        )
        last_training = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(DISTINCT domain) FROM crime_records WHERE domain IS NOT NULL AND domain != ''"
        )
        total_domains = cursor.fetchone()[0]

        cursor.execute(
            "SELECT crime_category, SUM(case_count) as total FROM crime_records "
            "GROUP BY crime_category ORDER BY total DESC LIMIT 1"
        )
        top_crime = cursor.fetchone()
        top_crime_name = top_crime[0] if top_crime else None
        top_crime_count = top_crime[1] if top_crime else 0

        cursor.execute("SELECT SUM(case_count) FROM crime_records")
        total_cases = cursor.fetchone()[0] or 0

        cursor.execute(
            "SELECT COUNT(DISTINCT legal_section_bns) FROM crime_records "
            "WHERE legal_section_bns IS NOT NULL AND legal_section_bns != ''"
        )
        bns_sections = cursor.fetchone()[0]

        conn.close()

        return {
            "total_records": total_records,
            "total_categories": total_categories,
            "total_domains": total_domains,
            "min_year": min_year,
            "max_year": max_year,
            "latest_month": latest_month,
            "latest_year": latest_year,
            "trained_models": trained_models,
            "last_training": last_training,
            "top_crime_name": top_crime_name,
            "top_crime_count": top_crime_count,
            "total_cases": total_cases,
            "bns_sections": bns_sections,
            "current_version": self.get_current_version()
        }
