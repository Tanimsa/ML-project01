# ML-project01
# Explainable ML for Spatial-Temporal Groundwater Quality Modeling



## 📍 Anglian Catchment
An intermediate-level predictive machine learning pipeline using an **eXtreme Gradient Boosting (XGBoost)** regressor to map spatial-temporal nitrate concentrations ($NO_3^-$ as $N$) across the hydrogeologically sensitive Anglian Catchment, UK. Powered by the UK Environment Agency Data Services Platform, this project integrates **SHAP (SHapley Additive exPlanations)** to eliminate "black-box" limitations and isolate localized contamination drivers.

> **Strategic Framework Alignment:** Aligns with active strategic directives from the **UK Department for Environment, Food & Rural Affairs (Defra)** and Australian state frameworks for digital twin infrastructure mapping.

---

## 🛠️ Data & Feature Pipeline

* **Data Acquisition:** Programmatically compiled from the UK EA Water Quality Explorer portal targeting records up through 2026.
* **Target Filter:** Strictly isolated to groundwater matrices (`determinand.definition == 'Groundwater'`) and Nitrate analytics (`determinand.label == 'Nitrate as N'`).
* **Spatial Vectors:** Longitude ($X_1$) and Latitude ($X_2$).
* **Temporal Deconstruction:** Parsed `phenomenonTime` into Year ($X_3$), Month ($X_4$), and Day of Year ($X_5$).
* **Censored Data QA:** Stripped alphanumeric Minimum Detection Limit (MDL) text prefixes (e.g., `<0.196`) to preserve model mathematical compatibility.

```python
# Data QA Protocol for Censored Observations
df['result'] = df['result'].astype(str).str.replace('<', '', regex=False)
df['result'] = pd.to_numeric(df['result'], errors='coerce')
df = df.dropna(subset=['result'])
