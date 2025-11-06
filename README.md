# 🌍 Global Terrorism Analysis & Machine Learning Prediction Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-Academic-green)

## 📖 Project Overview
This project performs **Exploratory Data Analysis (EDA)** and **Machine Learning modeling** on the **Global Terrorism Database (GTD)** to uncover terrorism patterns and **predict the terrorist group** responsible for a given attack.

It analyzes **geographical spread, yearly patterns, attack methods, target types**, and uses ML to classify incidents.  
A **Streamlit dashboard** is provided for visualization and prediction.

Developed by: **Ayush Gupte (PRN: 22070521120 | Batch: 2022–26)**

---

## 🗂 Dataset Information
| Property | Details |
|---------|---------|
| **Dataset Name** | Global Terrorism Database (GTD) |
| **Timeline** | 1970–2020 |
| **Total Records** | 289,796 |
| **Original Columns** | 135 |
| **Final ML Columns** | 11 |

### 🔗 Dataset Sources
- Official Website: https://www.start.umd.edu/gtd-download (used in project)
- Kaggle Mirror: https://www.kaggle.com/datasets/START-UMD/gtd
- **Dataset Included in Repo:** `Original Dataset/` and `Cleaned Dataset/`

---

## ✅ Final Feature Set Used in ML
```
Year, Month, Country, Region, Latitude, Longitude,
AttackType, TargetType, WeaponType, Suicide, GroupName
```

---

## 📊 Exploratory Data Analysis (Main_EDA_3.ipynb)

### 1️⃣ Global Terrorism Spread
![Global Terrorism Map](Global_Terrorism_Incident_Map.png)

### 2️⃣ Terrorism in Indian Cities
![Terrorism in India Cities](India_cities_with_terrorist_attack.png)

### Key Findings
- Incidents rise significantly after **2000**, peaking in **2014**.
- **Middle East & South Asia** are the highest affected regions.
- **Bombing/Explosion** is the most common attack type.
- **Taliban & ISIL** dominate recent terrorism.
- Suicide attacks are rare but extremely lethal.

---

## 🧹 Data Cleaning & Processing (Data_Cleaning_2.ipynb)
| Step | Description |
|------|-------------|
| Drop noisy features | Removed irrelevant & high-null columns |
| Handle missing values | Mean/Mode imputation |
| Normalize text fields | Unified labels & categories |
| Save final dataset | Saved to `Cleaned Dataset/` |

---

## 🤖 Machine Learning Model (Main_ML_file.ipynb)
This notebook performs **EDA + Model Training**.

| Step | Description |
|------|-------------|
| Encoding | Used LabelEncoder on categorical features |
| Split | Train/Test = 80/20 |
| Algorithm Used | **Random Forest Classifier** |
| Evaluation | Accuracy + Classification Report |
| Saved Encoder | `attack_type_label_encoder.pkl` |

> Future Enhancements: XGBoost, LightGBM, ANN, Explainability via SHAP.

---

## 💻 Streamlit Dashboard (app.py)
Features:
- Interactive terrorism visual trends
- Predict terrorist group based on user inputs

### Run Application
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure
```
├── Original Dataset/
├── Cleaned Dataset/
├── Data_Cleaning_2.ipynb
├── Main_EDA_3.ipynb
├── Main_ML_file.ipynb
├── app.py
├── Global_Terrorism_Incident_Map.png
├── India_cities_with_terrorist_attack.png
├── global_terrorism_map.html
├── attack_type_label_encoder.pkl
└── README.md
```

---

## 🚀 Future Improvements
- Deploy Streamlit App to **Render / Railway / AWS**
- Add **LSTM sequence forecasting**
- Build **Real-time Terrorism Monitoring Dashboard**
- Integrate **SHAP Explainability for trustable AI**

---

## 👤 Author
**Ayush Gupte**  
GitHub: https://github.com/AyushGupte-22

---

## 📜 License
This project is for **academic and research purposes only**.  
Dataset © Global Terrorism Database (GTD).
