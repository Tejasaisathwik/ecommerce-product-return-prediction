# MLOps Internal Laboratory Examination Report

**Subject:** Machine Learning Operations (MLOps) Laboratory  
**Question Number:** Question 4  
**Question Title:** Develop and Automate an E-Commerce Product Return Prediction System using CI/CD  
**Student Name:** Tejasaisathwik  
**Date of Submission:** November 2026  
**Repository Link:** `https://github.com/Tejasaisathwik/ecommerce-product-return-prediction`  
**Application Deployment Link:** `http://localhost:8000/docs` (Swagger UI) / Docker GHCR Container Image

---

## 1. Problem Statement
An e-commerce enterprise experiences substantial revenue erosion and logistics complexity due to high product return rates. In conventional operations, returns are identified reactively after shipment and delivery, leading to elevated reverse logistics overhead, repackaging costs, and customer churn.

The objective is twofold:
1. **Machine Learning Development:** Build an end-to-end classification system to predict whether a purchased item will be **Returned** or **Not Returned** based on product category, product price, discount offered, delivery days (expected vs actual), pre-order customer ratings, prior customer purchase/return history, package condition, and promotional flags.
2. **MLOps CI/CD Automation:** Establish a continuous integration and continuous deployment (CI/CD) pipeline using GitHub Actions that automates dependency management, linting, unit testing, Docker image packaging, and publishing to the GitHub Container Registry (GHCR) whenever source code or configuration changes occur.

---

## 2. Tools and Technologies Used

| Tool / Technology | Version | Purpose in Pipeline |
| :--- | :--- | :--- |
| **Python** | `3.11 / 3.13` | Primary programming language |
| **Pandas & NumPy** | `3.0.6 / 2.5.3` | Data ingestion, sanitization, and matrix operations |
| **Scikit-Learn** | `1.9.1` | Preprocessing, feature transformation, model training & evaluation |
| **FastAPI** | `0.141.1` | High-performance asynchronous REST API framework for model serving |
| **Uvicorn** | `0.53.0` | Production ASGI web server |
| **Pydantic** | `2.13.5` | Strict data validation, request parsing, and OpenAPI schema generation |
| **Pytest** | `9.2.0` | Automated test suite execution (unit tests and endpoint tests) |
| **Flake8** | `7.5.0` | PEP 8 static code analysis and linting |
| **Docker** | `29.8.2` | Application containerization and reproducible deployment |
| **GitHub Actions** | Hosted CI/CD | Continuous integration, automated test runners, and multi-stage workflow |
| **GHCR (ghcr.io)** | Container Registry | Cloud container image hosting and versioned distribution |
| **Joblib** | `1.6.0` | Model artifact persistence and serialization |
| **Matplotlib & Seaborn** | `3.11.2 / 0.13.2` | Evaluation curve plotting and exploratory data analysis |

---

## 3. Implementation Steps & Task Mapping

### Task 1: Dataset Loading, Preprocessing & Invalid Value Handling
The dataset contains 1,200 transaction records with 14 attributes. During exploratory data analysis, real-world data quality issues were identified and systematically resolved:
1. **Invalid Product Prices:** `Product_Price_INR` contained negative values (e.g., `-200.0`). Retail prices cannot be negative. These were detected and converted to `NaN`, followed by median imputation.
2. **Invalid Discount Percentages:** `Discount_Percent` contained values exceeding 100% (e.g., `125.0%`). Discounts were constrained to $[0, 100]$; values outside were converted to `NaN` and imputed.
3. **Delivery Days Consistency:** Delivery days were sanitized to prevent negative values.
4. **Customer Return Logic:** If a customer's `Previous_Returns` exceeded their total `Previous_Orders`, the return count was clipped to `Previous_Orders`.
5. **Missing Value Imputation:**
   - Numerical columns (`Product_Price_INR`, `Discount_Percent`, `Actual_Delivery_Days`, `Customer_Rating_PreOrder`) imputed via median strategy to remain robust against skewness.
   - Categorical columns (`Payment_Method`) imputed via the most-frequent strategy.
6. **Categorical Encoding:** One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`) applied to `Product_Category`, `Payment_Method`, `Damaged_Package_Flag`, `Package_Size`, and `Promotion_Active`.

---

### Task 2: Domain Feature Engineering
Five domain-specific features were engineered to capture consumer behavior and logistics risk:
1. **Delivery Delay (`delivery_delay`):**  
   $$\text{delivery\_delay} = \text{Actual\_Delivery\_Days} - \text{Expected\_Delivery\_Days}$$
   Positive values denote delayed shipments; negative values denote early fulfillment.
2. **Is Delayed Flag (`is_delayed`):**  
   Binary flag ($\{0, 1\}$) indicating whether delivery exceeded the promised turnaround time.
3. **Customer Return Frequency (`customer_return_frequency`):**  
   $$\text{customer\_return\_frequency} = \frac{\text{Previous\_Returns}}{\max(\text{Previous\_Orders}, 1)}$$
   Measures historical return propensity normalized between $0.0$ and $1.0$.
4. **Discount Amount (`discount_amount`):**  
   $$\text{discount\_amount} = \text{Product\_Price\_INR} \times \left(\frac{\text{Discount\_Percent}}{100}\right)$$
   Quantifies absolute monetary discount value.
5. **Effective Paid Price (`effective_price`):**  
   $$\text{effective\_price} = \text{Product\_Price\_INR} - \text{discount\_amount}$$

---

### Task 3: Machine Learning Model Training & Comparative Evaluation
Three distinct classification models were trained on 80% stratified training data ($N=960$) and tested on 20% held-out test data ($N=240$). Because product returns represent an imbalanced target (~22.6% return rate), class balancing was incorporated.

#### Evaluation Metric Comparison

| Metric | Random Forest (Balanced) | Gradient Boosting | Logistic Regression (Balanced) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 66.25% | 75.42% | 60.83% |
| **Precision** | 27.87% | 27.27% | 27.27% |
| **Recall (Return Class)** | 31.48% | 5.56% | **44.44%** |
| **F1-Score** | 0.2957 | 0.0923 | **0.3380** |
| **ROC-AUC** | 0.5696 | 0.5488 | **0.5775** |

#### Best Model Selection Rationale
In reverse logistics, **Recall** for returned items is critical because unpredicted returns incur high unexpected freight and restocking costs. Balanced Logistic Regression achieved the highest ROC-AUC ($0.5775$), highest Recall ($44.44\%$), and highest F1-Score ($0.3380$), making it the optimal model for production deployment. The entire preprocessor and trained estimator were serialized into `models/best_model.joblib`.

---

### Task 4: FastAPI Application Development
A production-ready inference service was built using FastAPI in `app/main.py`:
- **GET `/health`**: Returns system status, loaded model name, and API readiness.
- **POST `/predict`**: Receives order payload, validates inputs with Pydantic, applies feature transformations, executes inference, and outputs prediction label (`Returned` / `Not Returned`), probability, and computed engineered features.
- **POST `/batch-predict`**: Enables high-throughput batch scoring for bulk order streams.
- **GET `/metrics`**: Exposes model validation metrics and confusion matrix metadata.
- **Swagger UI (`/docs`) & ReDoc (`/redoc`)**: Interactive documentation for seamless verification.

---

### Task 5: GitHub Repository Structure & Configuration
The repository includes complete code, trained models, test scripts, Docker configuration, and CI/CD pipelines structured cleanly according to industry standards.

---

### Task 6: Automated CI/CD Pipeline with GitHub Actions
Implemented in `.github/workflows/ci-cd.yml`:
1. **Lint & Test Stage:**
   - Runs on `ubuntu-latest` with Python 3.11.
   - Executes `flake8` for style and syntax checks.
   - Executes `pytest` covering 8 unit and integration tests.
   - Publishes test reports as downloadable workflow artifacts.
2. **Build & Publish Stage:**
   - Triggers automatically upon successful test passage.
   - Configures Docker Buildx.
   - Authenticates to GitHub Container Registry using `secrets.GITHUB_TOKEN`.
   - Builds optimized container image with caching.
   - Pushes image to `ghcr.io/tejasaisathwik/ecommerce-product-return-prediction:latest`.
3. **Smoke Test Stage:**
   - Pulls published image from GHCR.
   - Runs container and queries `/health` endpoint to verify live execution.

---

### Task 7: Containerization & GHCR Publication
A production `Dockerfile` was authored using `python:3.11-slim`:
- Non-root lightweight execution.
- Native `HEALTHCHECK` verifying API responsiveness.
- Multi-worker Uvicorn startup.
- Published to: `ghcr.io/tejasaisathwik/ecommerce-product-return-prediction:latest`.

---

### Task 8 & 9: Triggering Workflow, Inspecting Build Logs & Demonstration
1. When changes are made to application code or model parameters, committing and pushing to `main` automatically triggers the GitHub Actions workflow.
2. All jobs (`lint-and-test`, `build-and-publish`, `smoke-test-container`) run in sequence.
3. Build logs can be inspected under the **Actions** tab of the GitHub repository.
4. The newly generated Docker image can be pulled and run anywhere using standard Docker CLI commands.

---

## 4. Execution & Faculty Verification Guide

### Step 1: Run Unit Tests and Linting
```bash
python -m flake8
python -m pytest -v
```
**Expected Result:** 8 passed tests, 0 linting errors.

### Step 2: Start FastAPI Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/docs` in a browser.

### Step 3: Test Health Endpoint
```bash
curl http://localhost:8000/health
```
**Output:**
```json
{"status":"healthy","model_loaded":true,"model_name":"LogisticRegression","app_version":"1.0.0"}
```

### Step 4: Test Sample Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Product_Category": "Fashion",
       "Product_Price_INR": 12825.0,
       "Discount_Percent": 25.0,
       "Expected_Delivery_Days": 5,
       "Actual_Delivery_Days": 8,
       "Customer_Rating_PreOrder": 3.8,
       "Previous_Returns": 3,
       "Previous_Orders": 10,
       "Payment_Method": "COD",
       "Damaged_Package_Flag": "No",
       "Package_Size": "Medium",
       "Promotion_Active": "Yes"
     }'
```
**Output:**
```json
{
  "prediction": "Returned",
  "is_returned": true,
  "return_probability": 0.7279,
  "model_name": "LogisticRegression",
  "engineered_features": {
    "delivery_delay_days": 3.0,
    "is_delayed": true,
    "customer_return_frequency": 0.3,
    "discount_amount_inr": 3206.25,
    "effective_price_inr": 9618.75
  }
}
```

### Step 5: Run with Docker
```bash
docker compose up --build -d
```

---

## 5. Summary of Results
- Successfully built end-to-end ML pipeline with domain feature engineering.
- Resolved real-world dataset anomalies including negative prices and discounts exceeding 100%.
- Trained and evaluated 3 classification models with balanced class weights.
- Deployed REST API via FastAPI with Pydantic schema validation.
- Configured automated GitHub Actions CI/CD pipeline building Docker images and publishing to GHCR.
- Fully verified with 100% test pass rate and production container readiness.
