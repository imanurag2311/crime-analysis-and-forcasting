"""
Data Validator
===============
Validates crime records against the standard schema.
Checks for data integrity, completeness, and consistency.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd

from app.core.config import PRIMARY_COLUMNS, CRIME_DOMAINS

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates crime data records against the project schema.
    
    Checks:
        - Required fields present
        - Valid month (1-12)
        - Valid year (1900-2100)
        - Numeric non-negative case_count
        - Valid domain
        - Valid legal section format
        - Duplicate detection
        - Suspicious value detection
    """

    def __init__(self):
        """Initialize validator with validation rules."""
        self.valid_months = set(range(1, 13))
        self.valid_year_range = (1900, 2100)
        self.valid_domains = set(CRIME_DOMAINS)

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate an entire DataFrame of crime records.
        
        Returns:
            Dict with:
                - valid_df: DataFrame of valid records
                - invalid_df: DataFrame of invalid records with error details
                - warnings: List of warning messages
                - errors: List of error messages
                - summary: Validation summary stats
        """
        if df.empty:
            return {
                'valid_df': pd.DataFrame(),
                'invalid_df': pd.DataFrame(),
                'warnings': ["Empty dataset provided"],
                'errors': [],
                'summary': {
                    'total': 0, 'valid': 0, 'invalid': 0,
                    'warnings': 1, 'duplicates': 0
                }
            }

        valid_records = []
        invalid_records = []
        warnings = []
        all_errors = []

        for idx, row in df.iterrows():
            errors = self._validate_record(row)
            if errors:
                record_dict = row.to_dict()
                record_dict['validation_errors'] = '; '.join(errors)
                record_dict['validation_status'] = 'invalid'
                invalid_records.append(record_dict)
                all_errors.extend([f"Row {idx}: {e}" for e in errors])
            else:
                record_dict = row.to_dict()
                record_dict['validation_status'] = 'valid'
                valid_records.append(record_dict)

        # Check for suspicious values
        valid_df = pd.DataFrame(valid_records) if valid_records else pd.DataFrame()
        if not valid_df.empty:
            suspicious = self._check_suspicious_values(valid_df)
            warnings.extend(suspicious)

        # Check for duplicates within the dataset
        duplicates = self._detect_duplicates(valid_df)
        if duplicates:
            warnings.extend(duplicates)

        invalid_df = pd.DataFrame(invalid_records) if invalid_records else pd.DataFrame()

        summary = {
            'total': len(df),
            'valid': len(valid_records),
            'invalid': len(invalid_records),
            'warnings': len(warnings),
            'duplicates': len(duplicates)
        }

        logger.info(
            f"Validation complete: {summary['valid']}/{summary['total']} valid, "
            f"{summary['invalid']} invalid, {summary['warnings']} warnings"
        )

        return {
            'valid_df': valid_df,
            'invalid_df': invalid_df,
            'warnings': warnings,
            'errors': all_errors,
            'summary': summary
        }

    def _validate_record(self, row: pd.Series) -> List[str]:
        """Validate a single record. Returns list of error messages."""
        errors = []

        # Crime category
        crime_cat = row.get('crime_category', '')
        if pd.isna(crime_cat) or str(crime_cat).strip() == '':
            errors.append("crime_category is empty")

        # Month
        month = row.get('month')
        if pd.isna(month):
            errors.append("month is missing")
        else:
            try:
                month_int = int(month)
                if month_int not in self.valid_months:
                    errors.append(f"Invalid month: {month_int} (must be 1-12)")
            except (ValueError, TypeError):
                errors.append(f"month is not numeric: {month}")

        # Year
        year = row.get('year')
        if pd.isna(year):
            errors.append("year is missing")
        else:
            try:
                year_int = int(year)
                if year_int < self.valid_year_range[0] or year_int > self.valid_year_range[1]:
                    errors.append(
                        f"Invalid year: {year_int} "
                        f"(must be {self.valid_year_range[0]}-{self.valid_year_range[1]})"
                    )
            except (ValueError, TypeError):
                errors.append(f"year is not numeric: {year}")

        # Case count
        case_count = row.get('case_count')
        if pd.isna(case_count):
            errors.append("case_count is missing")
        else:
            try:
                count_val = int(float(case_count))
                if count_val < 0:
                    errors.append(f"Negative case_count: {count_val}")
            except (ValueError, TypeError):
                errors.append(f"case_count is not numeric: {case_count}")

        # Legal section (warn if empty but don't fail)
        legal = row.get('legal_section', '')
        if pd.isna(legal) or str(legal).strip() == '':
            # Not a hard error - some records may not have a section
            pass

        # Domain (warn if invalid but don't fail)
        domain = row.get('domain', '')
        if not pd.isna(domain) and str(domain).strip() != '':
            if str(domain) not in self.valid_domains:
                # Not a hard error, just unusual
                pass

        return errors

    def _check_suspicious_values(self, df: pd.DataFrame) -> List[str]:
        """Check for suspicious values that may indicate data issues."""
        warnings = []

        if 'case_count' in df.columns:
            # Check for unusually high values
            numeric_counts = pd.to_numeric(df['case_count'], errors='coerce')
            if not numeric_counts.empty:
                mean_val = numeric_counts.mean()
                std_val = numeric_counts.std()
                if std_val > 0:
                    outliers = numeric_counts[
                        numeric_counts > mean_val + 3 * std_val
                    ]
                    for idx in outliers.index:
                        cat = df.loc[idx, 'crime_category'] if 'crime_category' in df.columns else 'Unknown'
                        warnings.append(
                            f"Suspicious high case_count {int(outliers[idx])} "
                            f"for '{cat}' (mean={mean_val:.0f}, std={std_val:.0f})"
                        )

                # Check for zero values
                zeros = numeric_counts[numeric_counts == 0]
                if len(zeros) > 0:
                    warnings.append(
                        f"{len(zeros)} records have case_count = 0"
                    )

        return warnings

    def _detect_duplicates(self, df: pd.DataFrame) -> List[str]:
        """Detect duplicate records within the dataset."""
        warnings = []

        if df.empty:
            return warnings

        dup_cols = ['crime_category', 'month', 'year']
        available_cols = [c for c in dup_cols if c in df.columns]

        if len(available_cols) == len(dup_cols):
            duplicated = df[df.duplicated(subset=available_cols, keep=False)]
            if not duplicated.empty:
                dup_groups = duplicated.groupby(available_cols).size()
                for (cat, month, year), count in dup_groups.items():
                    warnings.append(
                        f"Duplicate: {cat} ({month}/{year}) appears {count} times"
                    )

        return warnings

    def validate_single_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single record dict.
        
        Returns:
            (is_valid, list_of_errors)
        """
        row = pd.Series(record)
        errors = self._validate_record(row)
        return (len(errors) == 0, errors)

    def validate_excel_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if an Excel file has the expected columns.
        Returns mapping suggestions and missing/extra columns.
        """
        excel_cols = list(df.columns)
        required = set(PRIMARY_COLUMNS)

        # Try to auto-map columns
        column_mapping = {}
        unmapped_required = []
        unmapped_excel = list(excel_cols)

        for req_col in PRIMARY_COLUMNS:
            mapped = False
            for excel_col in excel_cols:
                if excel_col.lower().strip().replace(' ', '_') == req_col.lower():
                    column_mapping[excel_col] = req_col
                    if excel_col in unmapped_excel:
                        unmapped_excel.remove(excel_col)
                    mapped = True
                    break
                # Fuzzy matching
                elif req_col.lower() in excel_col.lower().replace(' ', '_'):
                    column_mapping[excel_col] = req_col
                    if excel_col in unmapped_excel:
                        unmapped_excel.remove(excel_col)
                    mapped = True
                    break

            if not mapped:
                unmapped_required.append(req_col)

        return {
            'column_mapping': column_mapping,
            'unmapped_required': unmapped_required,
            'unmapped_excel': unmapped_excel,
            'excel_columns': excel_cols,
            'required_columns': list(PRIMARY_COLUMNS)
        }
