# 🏆 Evaluate Classification ML
**An AutoML-lite Marathon Engine for Scikit-Learn Classifiers**

Developed by **BELBIN BENO RM**

[cite_start]Evaluate Classification is a robust framework designed to automate the evaluation of over 30+ classification models simultaneously[cite: 3, 9]. [cite_start]It handles everything from simple Linear models to "Boosting Giants" like XGBoost and CatBoost, while ensuring your system stays stable through active RAM and Time guarding[cite: 3, 17, 21].

---

## ✨ Key Features

* [cite_start]**🏎️ Classification Marathon:** Automatically trains and evaluates a massive catalog including Boosting, Trees, Ensembles, SVMs, and Neural Networks[cite: 3, 4, 8].
* [cite_start]**🛡️ Resource Guarding:** Features an "Active Kill" system that terminates processes exceeding a 900s time limit or 10GB RAM threshold to prevent system crashes[cite: 9, 17, 21].
* [cite_start]**💾 Performance Persistence:** Automatically saves every successful model as a `.joblib` file named with its metrics: `Model_TrainAcc_ValAcc_ValF1_Time_RAM.joblib`[cite: 13, 26].
* [cite_start]**📊 Live Registry:** Maintains a real-time `score_df` (pandas DataFrame) sorted by Validation Accuracy for instant comparison[cite: 16, 17].
* [cite_start]**🛠️ Lifecycle Helpers:** Built-in methods for model inspection, directory cleanup, and zipping results for deployment[cite: 10, 11].

---

## 🚀 Quick Start

### Notebook Installation (Kaggle / Colab / Jupyter)
```python
!pip install -q git+https://github.com/BELBINBENORM/evaluate-classification-ml.git
```
---
## 🚀 Basic Usage

```python
from Evaluate_Classification import EvaluateClassification

# 1. Initialize the engine
eval_cls = EvaluateClassification() [cite: 10]

# 2. Run the Marathon
# Fits and saves models not already present in your directory
eval_cls.evaluate(X_train, X_val, y_train, y_val) [cite: 10, 18]

# 3. View the Leaderboard
df = eval_cls.score() [cite: 10, 16]
print(df.head())

# 4. Inspect a specific model
eval_cls.inspection("XGBoost") [cite: 11, 32]
```
---
## 📊 Evaluation Output
[cite_start]When calling `eval_cls.score()`, you get a detailed snapshot of your model leaderboard, automatically sorted by the best Validation Accuracy[cite: 16]:

| Model | Group | Train_Acc | Val_Acc | Val_F1 | Time_Sec | RAM_GB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | Boosting | 0.9921 | 0.9450 | 0.9410 | 12.4 | 0.150 |
| **RandomForest** | Ensemble | 1.0000 | 0.9320 | 0.9280 | 5.2 | 0.082 |
| **LightGBM** | Boosting | 0.9850 | 0.9210 | 0.9150 | 8.1 | 0.110 |

---

## 💡 Why use Evaluate Classification?
The primary advantage of this framework is its robust **Multi-Process Architecture** and automated model management:

* [cite_start]**🛠️ Integrated Catalog:** Contains a pre-configured library of over 30 classifiers including Boosting Giants, SVMs, and Neural Networks.
* [cite_start]**🛡️ Active Resource Guarding:** Each model runs in an isolated process with a 15-minute time limit and a 10GB RAM threshold to prevent environment crashes[cite: 9, 17, 21, 22].
* [cite_start]**🔄 Smart Resumption:** The engine automatically scans for existing `.joblib` files and skips models already present in the directory to save time[cite: 13, 18, 19].
* [cite_start]**📦 Persistence & Portability:** Models are saved with performance metrics in the filename and can be instantly zipped for deployment[cite: 26, 36, 37].

---
## 📬 Contact

[cite_start]**Author:** BELBIN BENO RM [cite: 1]  
[cite_start]**Email:** [belbin.datascientist@gmail.com](mailto:belbin.datascientist@gmail.com) [cite: 1]  
[cite_start]**GitHub:** [BELBINBENORM](https://github.com/BELBINBENORM) [cite: 1]

