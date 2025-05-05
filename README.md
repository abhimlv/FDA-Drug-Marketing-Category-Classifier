# 💊 FDA Product Classification using Machine Learning and Deep Learning Models

This project presents a robust **multi-class classification pipeline** to predict the **marketing category** of FDA-listed pharmaceutical products. Leveraging a blend of classical ML models and deep learning, the pipeline handles structured metadata to categorize each product into NDA, ANDA, OTC, BLA, or UNAPPROVED.

---

## 📌 Project Highlights

* **Task**: Multi-class classification of FDA drug entries by marketing category
* **Input Data**: Structured metadata (product name, route, substance name, year, etc.)
* **Models Used**: Logistic Regression, Random Forest, XGBoost, Deep Learning (Keras)
* **Techniques**: Label Encoding, Normalization, Hyperparameter Tuning, Embedding Layers
* **Evaluation**: Confusion Matrix, ROC Curve, Accuracy, AUC
* **Visuals**: SHAP values, Feature Importance, Distribution plots

---

## 🧪 Dataset

* **Source**: Public FDA database (processed version available below)
* 📂 [Dataset on Hugging Face](https://huggingface.co/datasets/abhimlv/FDA_Product_data/tree/main)

---

## 🛠️ Preprocessing Overview

The dataset includes text, categorical, and numeric fields. Preprocessing included:

* **Grouping categories** (e.g., combining subtypes under main classes like NDA)
* **Label Encoding** of categorical fields (e.g., route, substance name)
* **Normalization** of numeric features (year, name lengths)
* **Missing value handling** and visualization:

![Missing Values Heatmap](outputs/preprocessing/missing_value_heatmap.jpg)

---

## ✨ Feature Engineering

* Categorical variables encoded using `LabelEncoder`
* Numerical features normalized using mean-std scaling
* Feature importance visualized using SHAP and built-in model metrics:

![SHAP Feature Importance](outputs/feature_importance/shap_feature_importance.jpg)
![Random Forest Importance](outputs/feature_importance/rf_feature_importance.jpg)

---

## 🎯 Model Training

### Logistic Regression

* Baseline model for reference

![Confusion Matrix](outputs/confusion_matrices/lr_confusion_matrix.jpg)
![ROC Curve](outputs/roc_curves/logistic_regression_roc_curves.jpg)

### Random Forest

* Tuned using GridSearchCV

![Confusion Matrix](outputs/confusion_matrices/rf_confusion_matrix.jpg)
![ROC Curve](outputs/roc_curves/random_forest_roc_curves.jpg)

### XGBoost Classifier

* Strong performance, especially for BLA and NDA

![Confusion Matrix](outputs/confusion_matrices/xgb_confusion_matrix.jpg)
![ROC Curve](outputs/roc_curves/xgboost_roc_curves.jpg)

### Deep Learning (Keras)

* Initial dense model + improved model with Embedding layers

![Training History (Basic)](outputs/dl_training_history.jpg)
![Confusion Matrix](outputs/confusion_matrices/dl_confusion_matrix.jpg)

### Improved Deep Learning

* Uses embedding layers, tuned architecture, better generalization

![Training History](outputs/dl_improved_training_history.jpg)
![Confusion Matrix](outputs/confusion_matrices/dl_improved_confusion_matrix.jpg)
![ROC Curve](outputs/roc_curves/deep_learning_improved_roc_curves.jpg)

---

## 📊 Distribution Plots

![Original Category Distribution](outputs/distributions/original_category_distribution.jpg)
![Grouped Category Distribution](outputs/distributions/grouped_category_distribution.jpg)
![Year-wise Distribution](outputs/distributions/year_wise_distribution.jpg)

---

## 📁 Model & Dataset Access

* 📦 [Dataset on Hugging Face](https://huggingface.co/datasets/abhimlv/FDA_Product_data/tree/main)
* 🤖 [All Trained Models on Hugging Face](https://huggingface.co/abhimlv/FDA_Classification/tree/main)

---

## 📈 Evaluation Summary

| Model               | Accuracy | ROC AUC | Notes                       |
| ------------------- | -------- | ------- | --------------------------- |
| Logistic Regression | \~0.83   | 0.81    | Baseline                    |
| Random Forest       | \~0.96   | 0.97    | Best for UNAPPROVED/NDA     |
| XGBoost             | \~0.97   | 0.98    | Consistently strong         |
| Deep Learning       | \~0.96   | 0.97    | Good with tuning            |
| DL (Improved)       | \~0.99+  | 0.99+   | Best overall generalization |

---

## 🔍 Key Takeaways & Future Work

* Deep learning outperformed traditional ML after sufficient tuning
* Category imbalance addressed via grouping; future work can explore SMOTE
* SHAP revealed strong impact of route and proprietary name features

**Future Directions:**

* Experiment with BERT-style tabular encoders or TabNet
* Integrate external drug description texts for enrichment
* Add explainability dashboard (Gradio or Streamlit)

---

## 🧠 Summary

This project showcases a complete, reproducible pipeline for real-world regulatory data classification using both traditional ML and deep learning. It highlights preprocessing, feature importance, visual evaluation, and production-ready model deployment.

🎯 **Deployed models and data are hosted on Hugging Face for public access and further experimentation.**
📌 [Ready Tensor Publication](https://huggingface.co/datasets/abhimlv/FDA_Product_data/tree/main)
