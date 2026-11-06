"""Generate professional Word Document (.docx) for MLOps Internal Lab Examination."""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def set_cell_background(cell, fill_hex):
    """Set cell background color."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tc_pr.append(shd)


def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    """Set inner cell padding."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def add_code_box(doc, text):
    """Render a shaded code/output block."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_background(cell, "F4F6F8")
    set_cell_margins(cell, 80, 80, 120, 120)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x24, 0x29, 0x2E)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)



def create_report():
    doc = Document()

    # Set page margins to 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Styles
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # ---------------- HEADER / TITLE BLOCK ----------------
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_inst = title.add_run("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING\n")
    run_inst.font.size = Pt(13)
    run_inst.font.bold = True
    run_inst.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    run_sub = title.add_run("MLOps Internal Laboratory Examination Report\n")
    run_sub.font.size = Pt(16)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(0x00, 0x56, 0xB3)

    run_q = title.add_run("Question 4: Develop and Automate an E-Commerce Product Return Prediction System using CI/CD\n")
    run_q.font.size = Pt(12)
    run_q.font.italic = True
    run_q.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    # Student Details Box Table
    details_table = doc.add_table(rows=7, cols=2)
    details_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    details_table.autofit = False

    details_data = [
        ("Student Name:", "Teja Sai Sathwik"),
        ("Register Number:", "231FA04211"),
        ("GitHub Repository Link:", "https://github.com/Tejasaisathwik/ecommerce-product-return-prediction"),
        ("Public Live Swagger UI (Deployment):", "https://proven-tests-announcement-maryland.trycloudflare.com/docs"),
        ("Localhost Swagger UI Link:", "http://localhost:8001/docs"),
        ("Docker Hub Container Image:", "teja211/ecommerce-product-return-prediction:latest"),
        ("GitHub Container Registry (GHCR):", "ghcr.io/tejasaisathwik/ecommerce-product-return-prediction:latest"),
    ]

    for i, (k, v) in enumerate(details_data):
        row = details_table.rows[i]
        cell_k, cell_v = row.cells[0], row.cells[1]
        cell_k.width = Inches(2.5)
        cell_v.width = Inches(4.0)

        set_cell_background(cell_k, "F0F4F8")
        set_cell_background(cell_v, "FFFFFF")
        set_cell_margins(cell_k, 80, 80, 120, 120)
        set_cell_margins(cell_v, 80, 80, 120, 120)

        p_k = cell_k.paragraphs[0]
        run_k = p_k.add_run(k)
        run_k.font.bold = True
        run_k.font.size = Pt(10)
        run_k.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

        p_v = cell_v.paragraphs[0]
        run_v = p_v.add_run(v)
        run_v.font.size = Pt(10)
        if "http" in v or "ghcr" in v:
            run_v.font.color.rgb = RGBColor(0x00, 0x66, 0xCC)
            run_v.font.underline = True

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ---------------- 1. PROBLEM STATEMENT ----------------
    h1 = doc.add_heading("1. Problem Statement & System Objectives", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p = doc.add_paragraph(
        "In modern e-commerce enterprises, reverse logistics and unexpected product returns incur massive "
        "operational overhead, restocking costs, and customer churn. Traditional retail operations identify returns "
        "only reactively after packages have been shipped, received, and initiated for return by buyers. "
        "The primary goal of this project is to build an automated, production-grade Machine Learning classification "
        "and continuous deployment pipeline that predicts return probability before fulfillment and automates "
        "testing, container building, and deployment via CI/CD workflows."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "Key Project Objectives:\n"
        "1. Ingestion & Preprocessing: Handle missing values, sanitize domain-invalid values (negative prices, discounts > 100%), and perform categorical encoding.\n"
        "2. Feature Engineering: Create impactful domain features including delivery delay, customer return frequency, and discount amounts.\n"
        "3. Model Training & Benchmarking: Train and evaluate multiple classification models (Random Forest, Gradient Boosting, Logistic Regression) under class-imbalanced conditions.\n"
        "4. Model Serving: Develop a robust REST API using FastAPI serving the optimal model via /predict and /health endpoints.\n"
        "5. Automated CI/CD Pipeline: Construct GitHub Actions workflow for linting, automated unit testing, and Docker container packaging.\n"
        "6. Container Registry Publishing: Build and publish container images to GitHub Container Registry (GHCR) and Docker Hub.\n"
        "7. Code Change Automation: Demonstrate automatic pipeline triggering upon committing and pushing source code updates."
    )

    # ---------------- 2. TOOLS AND TECHNOLOGIES ----------------
    h1 = doc.add_heading("2. Tools & Technologies Utilized", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    tools_data = [
        ("Tool / Framework", "Version", "Role in System Architecture"),
        ("Python", "3.11 / 3.13", "Core programming runtime for data science and API service"),
        ("Pandas & NumPy", "3.0.6 / 2.5.3", "Data cleaning, matrix manipulation, feature calculations"),
        ("Scikit-Learn", "1.9.1", "Data preprocessing, column transformation, model training & metrics"),
        ("FastAPI", "0.141.1", "High-performance async REST API framework with Swagger OpenAPI"),
        ("Uvicorn", "0.53.0", "Production ASGI web server running FastAPI application"),
        ("Pydantic", "2.13.5", "Data validation schemas and request/response typing"),
        ("Pytest", "9.2.0", "Automated unit and integration test suite runner"),
        ("Flake8", "7.5.0", "PEP 8 static code quality analysis and syntax linting"),
        ("Docker", "29.8.2", "Containerization engine, Dockerfile, and multi-container orchestration"),
        ("GitHub Actions", "Hosted CI/CD", "Continuous integration and deployment workflow automation"),
        ("GHCR & Docker Hub", "Cloud Registries", "Versioned container image hosting and distribution"),
        ("Joblib", "1.6.0", "Model artifact serialization and deserialization"),
    ]

    t_tools = doc.add_table(rows=len(tools_data), cols=3)
    t_tools.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_tools.autofit = False

    for r_idx, row_data in enumerate(tools_data):
        row = t_tools.rows[r_idx]
        for c_idx, text in enumerate(row_data):
            cell = row.cells[c_idx]
            if c_idx == 0:
                cell.width = Inches(1.8)
            elif c_idx == 1:
                cell.width = Inches(1.2)
            else:
                cell.width = Inches(3.5)

            set_cell_margins(cell, 60, 60, 100, 100)
            p = cell.paragraphs[0]
            run = p.add_run(text)
            if r_idx == 0:
                set_cell_background(cell, "1B365D")
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(10)
            else:
                set_cell_background(cell, "F9FAFB" if r_idx % 2 == 1 else "FFFFFF")
                run.font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ---------------- 3. PREPROCESSING & FEATURE ENGINEERING ----------------
    h1 = doc.add_heading("3. Data Preprocessing & Feature Engineering (Tasks 1 & 2)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "The raw dataset consists of 1,200 transaction records with 14 attributes. "
        "During Exploratory Data Analysis, critical anomalies and invalid values were detected and sanitized:"
    )

    doc.add_paragraph(
        "1. Invalid Product Prices: Negative prices (e.g. -200.0 INR) were identified. Prices must be strictly positive; invalid entries were replaced with NaN and imputed using median.\n"
        "2. Invalid Discount Percentages: Values exceeding 100% (e.g. 125.0%) were detected. Values outside [0, 100] were marked as NaN and imputed.\n"
        "3. Illogical Return Counts: Instances where Previous_Returns exceeded Previous_Orders were capped at Previous_Orders to ensure mathematical consistency.\n"
        "4. Missing Value Imputation: Numerical features imputed with SimpleImputer(strategy='median') to resist outlier skew; categorical features imputed with SimpleImputer(strategy='most_frequent').\n"
        "5. Categorical Encoding: Scikit-learn OneHotEncoder(handle_unknown='ignore', sparse_output=False) applied to Product_Category, Payment_Method, Damaged_Package_Flag, Package_Size, and Promotion_Active."
    )

    doc.add_heading("Domain Feature Engineering Formulations:", level=2)
    doc.add_paragraph(
        "• Delivery Delay (days): Actual_Delivery_Days - Expected_Delivery_Days (positive values indicate tardiness)\n"
        "• Delayed Shipment Flag: Binary indicator: 1 if Delivery Delay > 0, else 0\n"
        "• Customer Return Frequency: Previous_Returns / max(Previous_Orders, 1) (normalized past return propensity)\n"
        "• Monetary Discount Amount (INR): Product_Price_INR * (Discount_Percent / 100.0)\n"
        "• Effective Paid Price (INR): Product_Price_INR - Discount_Amount"
    )

    # ---------------- 4. MODEL BENCHMARK & EVALUATION ----------------
    h1 = doc.add_heading("4. Machine Learning Model Training & Comparative Evaluation (Task 3)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "Three distinct classification models were trained on 80% training data (960 samples) and evaluated on a "
        "20% held-out stratified test set (240 samples). Because returned orders constitute an imbalanced class (~22.6%), "
        "class weighting and probability thresholds were rigorously analyzed."
    )

    metrics_table_data = [
        ("Classifier Model", "Accuracy", "Precision", "Recall (Returns)", "F1-Score", "ROC-AUC", "Status"),
        ("Logistic Regression (Balanced)", "60.83%", "27.27%", "44.44%", "0.3380", "0.5775", "Selected Best Model"),
        ("Random Forest (Balanced)", "66.25%", "27.87%", "31.48%", "0.2957", "0.5696", "Candidate Model 2"),
        ("Gradient Boosting Classifier", "75.42%", "27.27%", "5.56%", "0.0923", "0.5488", "Candidate Model 3"),
    ]

    t_metrics = doc.add_table(rows=len(metrics_table_data), cols=7)
    t_metrics.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_metrics.autofit = False

    for r_idx, row_data in enumerate(metrics_table_data):
        row = t_metrics.rows[r_idx]
        for c_idx, text in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            run = p.add_run(text)
            if r_idx == 0:
                set_cell_background(cell, "1B365D")
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9.5)
            else:
                set_cell_background(cell, "EBF5FB" if r_idx == 1 else "FFFFFF")
                run.font.size = Pt(9)
                if r_idx == 1:
                    run.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "Selection Rationale: In reverse logistics management, failing to predict a return (false negative) is far more "
        "costly than flagging an order for inspection. Balanced Logistic Regression achieved the highest Recall (44.44%), "
        "highest F1-Score (0.3380), and highest ROC-AUC (0.5775), and was serialized into models/best_model.joblib."
    )

    # Insert Model Evaluation Visualizations
    doc.add_heading("Model Evaluation Curves & Confusion Matrix:", level=2)
    for img_name, caption in [
        ("models/model_comparison.png", "Figure 1: Comprehensive Model Performance Comparison Across Metrics"),
        ("models/confusion_matrix.png", "Figure 2: Confusion Matrix for the Selected Best Model"),
        ("models/roc_curve.png", "Figure 3: Receiver Operating Characteristic (ROC) Curves Comparison"),
    ]:
        if os.path.exists(img_name):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            doc.add_picture(img_name, width=Inches(5.0))
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_cap = p_cap.add_run(caption)
            run_cap.font.italic = True
            run_cap.font.size = Pt(9.5)
            run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            p_cap.paragraph_format.space_after = Pt(12)

    # ---------------- 5. FASTAPI APPLICATION ----------------
    h1 = doc.add_heading("5. FastAPI Model Serving Implementation (Task 4)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "The model is deployed as a modular, high-throughput REST API using FastAPI (app/main.py):\n"
        "• GET /health: Health probe endpoint returning service status, loaded model name, and deployment metadata.\n"
        "• POST /predict: Accepts JSON payload with order attributes, automatically computes domain features, executes the trained pipeline, and returns prediction label, return probability, and risk tier.\n"
        "• POST /batch-predict: Bulk inference endpoint for high-volume transactions.\n"
        "• GET /metrics: Returns offline model evaluation metrics and confusion matrix metadata.\n"
        "• Swagger UI (/docs): Interactive OpenAPI documentation interface. Accessible publicly worldwide at https://proven-tests-announcement-maryland.trycloudflare.com/docs and locally at http://localhost:8001/docs."
    )

    # Insert Swagger UI screenshots
    for img_name, caption in [
        ("outputs/WhatsApp Image 2026-11-06 at 3.25.29 PM.jpeg", "Figure 4: FastAPI Interactive Swagger UI Documentation (Public: https://proven-tests-announcement-maryland.trycloudflare.com/docs / Local: http://localhost:8001/docs)"),
        ("outputs/WhatsApp Image 2026-11-06 at 3.25.39 PM.jpeg", "Figure 5: Pydantic v2 Request & Response Schema Definitions in Swagger UI"),
        ("outputs/WhatsApp Image 2026-11-06 at 3.25.53 PM.jpeg", "Figure 6: Live API Root Endpoint JSON Response (http://localhost:8001/)"),
    ]:
        if os.path.exists(img_name):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_picture(img_name, width=Inches(5.6))
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_cap = p_cap.add_run(caption)
            run_cap.font.italic = True
            run_cap.font.size = Pt(9.5)
            run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            p_cap.paragraph_format.space_after = Pt(12)

    # ---------------- 6. GITHUB REPOSITORY ----------------
    h1 = doc.add_heading("6. GitHub Repository Setup & Source Organization (Task 5)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "A public repository was established at https://github.com/Tejasaisathwik/ecommerce-product-return-prediction "
        "containing complete source code, configuration files, trained model artifact, datasets, Docker assets, "
        "and GitHub Actions workflows."
    )

    img_repo = "outputs/WhatsApp Image 2026-11-06 at 3.20.41 PM.jpeg"
    if os.path.exists(img_repo):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(img_repo, width=Inches(5.6))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 7: GitHub Repository Structure & Published GHCR Package")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p_cap.paragraph_format.space_after = Pt(12)

    # ---------------- 7. CI/CD PIPELINE ----------------
    h1 = doc.add_heading("7. Automated CI/CD Pipeline Implementation (Task 6)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "The automated CI/CD pipeline is configured in .github/workflows/ci-cd.yml and comprises three automated stages:\n"
        "1. Stage 1 - Continuous Integration (Lint & Test): Sets up Python 3.11, installs dependencies, executes flake8 linter (0 errors), executes pytest test suite (8/8 passed), and uploads test result artifacts.\n"
        "2. Stage 2 - Continuous Deployment (Docker Build & Push): Automatically builds an optimized multi-platform Docker container and publishes it to GitHub Container Registry (ghcr.io) authenticated via GITHUB_TOKEN.\n"
        "3. Stage 3 - Post-Deployment Smoke Test: Pulls the published container from GHCR, spins it up in an isolated runner environment, and validates the /health endpoint with cURL."
    )

    img_act_list = "outputs/WhatsApp Image 2026-11-06 at 3.21.15 PM.jpeg"
    if os.path.exists(img_act_list):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(img_act_list, width=Inches(5.6))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 8: GitHub Actions Workflow Execution History Showing All Successful Pipeline Runs")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p_cap.paragraph_format.space_after = Pt(12)

    img_run1 = "outputs/WhatsApp Image 2026-11-06 at 3.27.11 PM.jpeg"
    if os.path.exists(img_run1):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(img_run1, width=Inches(5.6))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 9: Detailed CI/CD Execution Graph (Run #1) Showing All 3 Jobs Completed Successfully")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p_cap.paragraph_format.space_after = Pt(12)

    # ---------------- 8. DOCKER & GHCR PUBLICATION ----------------
    h1 = doc.add_heading("8. Containerization & Registry Publication (Task 7)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "A production Dockerfile was constructed using python:3.11-slim. It installs curl, pip dependencies, "
        "copies source code and models, and embeds a native HEALTHCHECK. Images were published to both:\n"
        "• GitHub Container Registry: ghcr.io/tejasaisathwik/ecommerce-product-return-prediction:latest\n"
        "• Docker Hub Registry: teja211/ecommerce-product-return-prediction:latest"
    )

    img_doc_imgs = "outputs/WhatsApp Image 2026-11-06 at 3.26.11 PM.jpeg"
    if os.path.exists(img_doc_imgs):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(img_doc_imgs, width=Inches(5.6))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 10: Docker Desktop Images Dashboard Showing Built Images Tagged for Local and Docker Hub")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p_cap.paragraph_format.space_after = Pt(12)

    # ---------------- 9. TASK 8 CODE MODIFICATION ----------------
    h1 = doc.add_heading("9. Code Modification & Automatic Workflow Triggering (Task 8)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "To satisfy Task 8, functional code modifications were implemented in app/main.py, app/schemas.py, and tests/test_api.py:\n"
        "• Added CI/CD provenance metadata: ci_cd_pipeline: 'Automated via GitHub Actions & GHCR' and environment: 'production'\n"
        "• Bumped application release version to 1.2.0.\n"
        "• Committed with message: 'feat(api): expose ci_cd_pipeline and environment metadata, bump app version to 1.2.0' (Commit SHA: 76cdd2b).\n"
        "• Pushed to main branch, which automatically triggered GitHub Actions Workflow Run #42195088511."
    )

    img_run2 = "outputs/WhatsApp Image 2026-11-06 at 3.27.24 PM.jpeg"
    if os.path.exists(img_run2):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(img_run2, width=Inches(5.6))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 11: GitHub Actions Run #2 Triggered Automatically by Code Modification Commit (76cdd2b)")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p_cap.paragraph_format.space_after = Pt(12)

    # ---------------- 10. TASK 9 VERIFICATION ----------------
    h1 = doc.add_heading("10. Workflow Verification & Live Container Execution (Task 9)", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "Workflow build logs were inspected and confirmed zero errors. "
        "The updated Docker container was deployed locally on port 8001:8000 and verified live:"
    )

    img_cont = "outputs/WhatsApp Image 2026-11-06 at 3.26.22 PM.jpeg"
    if os.path.exists(img_cont):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(img_cont, width=Inches(5.6))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run("Figure 12: Docker Desktop Showing test_ecommerce_api Container Running Healthy on Port 8001:8000")
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p_cap.paragraph_format.space_after = Pt(12)

    doc.add_heading("Live Health Check Verification Output:", level=2)
    add_code_box(
        doc,
        '{\n'
        '  "status": "healthy",\n'
        '  "model_loaded": true,\n'
        '  "model_name": "LogisticRegression",\n'
        '  "app_version": "1.2.0",\n'
        '  "ci_cd_pipeline": "Automated via GitHub Actions & GHCR",\n'
        '  "environment": "production",\n'
        '  "timestamp": "2026-11-06T09:30:53.475307+00:00"\n'
        '}'
    )

    doc.add_heading("Live Single Order Prediction Output:", level=2)
    add_code_box(
        doc,
        '{\n'
        '  "prediction": "Returned",\n'
        '  "is_returned": true,\n'
        '  "return_probability": 0.5404,\n'
        '  "risk_level": "Moderate Risk",\n'
        '  "model_name": "LogisticRegression",\n'
        '  "engineered_features": {\n'
        '    "delivery_delay_days": 3.0,\n'
        '    "is_delayed": true,\n'
        '    "customer_return_frequency": 0.3,\n'
        '    "discount_amount_inr": 3206.25,\n'
        '    "effective_price_inr": 9618.75\n'
        '  },\n'
        '  "timestamp": "2026-11-06T09:02:48.590501+00:00"\n'
        '}'
    )

    # ---------------- 11. CONCLUSION ----------------
    h1 = doc.add_heading("11. Conclusion & Results Summary", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "All requirements specified for Question 4 have been successfully executed and validated:\n"
        "• Successfully preprocessed e-commerce dataset, addressing invalid negative prices and extreme discounts.\n"
        "• Developed domain feature engineering modules capturing shipment tardiness and customer return habits.\n"
        "• Trained and benchmarked 3 classification models; balanced Logistic Regression selected for superior return recall.\n"
        "• Created and verified production FastAPI service with public interactive Swagger UI documentation at https://proven-tests-announcement-maryland.trycloudflare.com/docs (and locally at http://localhost:8001/docs).\n"
        "• Implemented end-to-end GitHub Actions CI/CD pipeline building and publishing Docker images to GHCR and Docker Hub.\n"
        "• Verified automatic triggering upon pushing code modifications and demonstrated live execution of the updated container.\n"
        "• Packaged entire project into 231FA04211_MLOps_Internal.zip ready for Google Classroom submission."
    )

    output_path = "231FA04211_MLOps_Internal_Report.docx"
    doc.save(output_path)
    print(f"[+] Successfully generated Word document: {output_path}")
    print(f"    Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    return output_path


if __name__ == "__main__":
    create_report()
