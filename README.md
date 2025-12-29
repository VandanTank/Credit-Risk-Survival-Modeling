# Credit Risk Modeling with Survival Analysis: A Hierarchical Approach

## Project Overview

This project implements an advanced credit risk system that moves beyond traditional binary classification (predicting *if* a borrower defaults) to perform **Time-to-Event (Survival) Analysis**. The core objective is to estimate the *duration* until a loan is likely to default by calculating the **Cumulative Hazard of Default** over the loan's lifetime.

This analysis demonstrates a rigorous approach to predictive finance, featuring:
* The development of a non-linear, **hierarchical model**.
* A commitment to model **interpretability** using modern techniques.
* An audit of model predictions for **fairness** and ethical application.

**Project by:** Vandan Tank

---

## Final Output: Interactive Risk Predictor

The final artifact is an interactive web application built with Streamlit (`app.py`) that utilizes the trained survival model. Users can input borrower and loan parameters (Interest Rate, Grade, Term, etc.) and instantly receive a personalized **Survival Curve**.

The curve shows the predicted probability that the borrower will *not* default at any given month, allowing for precise, time-sensitive risk management.



## Key Findings and Insights

### 1. Model Efficiency and Rigor (Law of Diminishing Returns)

I rigorously tested the model's performance against computational cost by training on varying data subsets.

| Sample Size | Training Time | Concordance Index (C-Index) | Key Insight |
| :---: | :---: | :---: | :--- |
| **10% (106k Rows)** | 4.6 minutes | **0.6837** | **Champion Model:** Achieved the vast majority of predictive power with maximum efficiency. |
| **30% (320k Rows)** | 57.1 minutes | **0.6839** | **Diminishing Returns:** Proved that a 12x increase in compute time provided no meaningful gain in accuracy, justifying the 10% model as the most resource-efficient choice. |

### 2. Model Interpretability (The "Why")

Using **SHAP (SHapley Additive exPlanations)**, I opened the model's "black box" to confirm its logic.

| Rank | Feature | Interpretation |
| :---: | :--- | :--- |
| **#1** | `int_rate` | **Highest Impact:** The model correctly learned that the interest rate (the lender's original risk pricing) is the most powerful predictor. |
| **#2** | `term` | **High Impact:** The duration of the loan significantly affects risk (60-month loans are riskier than 36-month loans). |
| **#3** | `grade` | **High Impact:** The institutional credit grade (A, B, C...) is a critical signal. |
| **#16** | `purpose_small_business` | **Minimal Impact:** The model is not heavily biased by the loan's stated purpose. |

### 3. Ethical Fairness Audit

I conducted a synthetic test by comparing two **identical borrowers** whose only difference was their loan purpose (`Debt Consolidation` vs. `Small Business`). The survival curves were **nearly identical**, proving the model is **not biased** by the loan's stated purpose and adheres to ethical standards by relying on core financial metrics (`int_rate`, `dti`) instead.

---

## Technical Implementation Details

### Data

* **Dataset:** Lending Club Loan Data (accepted_2007_to_2018Q4.csv.gz)
* **Size:** 2.26 million raw records, filtered down to 1.33 million completed loans for training.

### Methodology

| Component | Tools Used | Function |
| :--- | :--- | :--- |
| **Model** | `sksurv.RandomSurvivalForest` | Implemented the non-linear "hierarchical model" for time-to-event prediction. |
| **Debugging** | `joblib.loky` backend | Forced a stable parallel processing environment to overcome memory/threading deadlocks on large data. |
| **EDA** | `lifelines.NelsonAalenFitter` | Calculated and visualized the **Cumulative Hazard of Default**. |

## Setup and Execution

1.  **Dependencies:** Ensure a Python environment is active (`venv`) and install requirements:
    ```bash
    pip install streamlit pandas numpy matplotlib joblib scikit-survival shap fastparquet
    ```
2.  **Data:** Place the raw data file (`accepted_2007_to_2018Q4.csv.gz`) into the `/data` folder.
3.  **Train:** Run all cells in the three notebooks sequentially to train the model and generate the `.joblib` files.
4.  **Launch App:** Execute the final application:
    ```bash
    streamlit run app.py
    ```


