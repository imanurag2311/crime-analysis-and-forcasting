# Project Use Case Diagrams (PlantUML / PlantText)

This folder contains all the official use case diagram code for the **Mumbai Crime Analysis & Forecasting System**.

All diagrams are pre-scaled and formatted in vertical portrait aspect ratio for clean printing and documentation reports, with clean academic headers.

---

### File Inventory

| File | Description | Target Scope |
| :--- | :--- | :--- |
| [`01_master_system_use_case.puml`](./01_master_system_use_case.puml) | **Master Use Case Diagram** | Comprehensive overview of all 5 system tabs, CLI orchestrator, and all actors. |
| [`02_analytics_and_trends_use_case.puml`](./02_analytics_and_trends_use_case.puml) | **Subsystem 1: Analytics & Trends (EDA)** | KPI cards, longitudinal trend analysis, monthly seasonality, detection rates, and export. |
| [`03_sarima_forecasting_use_case.puml`](./03_sarima_forecasting_use_case.puml) | **Subsystem 2: SARIMA Forecasting** | Forecast generation, 80%/95% confidence intervals, grid search retraining, metric evaluations, and residuals. |
| [`04_ipc_bns_crosswalk_use_case.puml`](./04_ipc_bns_crosswalk_use_case.puml) | **Subsystem 3: IPC ↔ BNS Crosswalk** | Section searching, offence comparison, penalty change tracking, and concordance export. |
| [`05_dataset_manager_use_case.puml`](./05_dataset_manager_use_case.puml) | **Subsystem 4: Monthly Dataset Manager** | File upload, schema validation, range checks, deduplication, versioning, rollback, and audit logging. |
| [`06_ocr_processing_hub_use_case.puml`](./06_ocr_processing_hub_use_case.puml) | **Subsystem 5: Computer Vision & OCR Hub** | Scanned PDF upload, PyMuPDF rendering, CLAHE/deskew preprocessing, EasyOCR row reconstruction, manual review, and ingestion. |
| [`07_cli_orchestration_use_case.puml`](./07_cli_orchestration_use_case.puml) | **Subsystem 6: CLI Orchestration** | CLI commands (`--dashboard`, `--pipeline`, `--import-excel`, `--train`, `--test`, `--status`). |

---

### How to Render in PlantText

1. Open **[PlantText](https://www.planttext.com/)** in any browser.
2. Open any `.puml` file above and copy its content.
3. Paste into the PlantText editor window and click **Refresh**.
4. Click **Export as PNG** or **Export as PDF** to save and print.
