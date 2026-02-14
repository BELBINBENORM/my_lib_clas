import os
import re
import gc
import joblib
import shutil
import time
import psutil
import multiprocessing
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, f1_score

# Ignore specific sklearn warnings for a cleaner console
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CLASSIFICATION IMPORTS ---
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, 
    Perceptron, PassiveAggressiveClassifier
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier,
    HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV

# Boosting Giants (Install via: pip install xgboost lightgbm catboost)
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
except ImportError:
    pass

def train_worker(model, X, y, return_dict):
    """Independent worker function for multiprocessing."""
    try:
        model.fit(X, y)
        return_dict['model'] = model
        return_dict['success'] = True
    except Exception as e:
        return_dict['success'] = False
        return_dict['error'] = str(e)
        
class EvaluateClassification:
    def __init__(self):
        # Full Classification Catalog
        self.catalog = {
            # Boosting Giants
            'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 'Boosting'),
            'LightGBM': (LGBMClassifier(verbose=-1), 'Boosting'),
            'CatBoost': (CatBoostClassifier(verbose=0), 'Boosting'),
            
            # Linear & GLM
            'LogisticRegression': (LogisticRegression(max_iter=1000), 'Linear'),
            'RidgeClassifier': (RidgeClassifier(), 'Linear'),
            'SGDClassifier': (SGDClassifier(), 'Linear'),
            'Perceptron': (Perceptron(), 'Linear'),
            'PassiveAggressive': (PassiveAggressiveClassifier(), 'Linear'),
            
            # Trees & Ensembles
            'DecisionTree': (DecisionTreeClassifier(), 'Tree'),
            'ExtraTree': (ExtraTreeClassifier(), 'Tree'),
            'RandomForest': (RandomForestClassifier(), 'Ensemble'),
            'ExtraTrees': (ExtraTreesClassifier(), 'Ensemble'),
            'Bagging': (BaggingClassifier(), 'Ensemble'),
            'GradientBoosting': (GradientBoostingClassifier(), 'Ensemble'),
            'HistGradientBoosting': (HistGradientBoostingClassifier(), 'Ensemble'),
            'AdaBoost': (AdaBoostClassifier(), 'Ensemble'),
            
            # Naive Bayes & Discriminant
            'GaussianNB': (GaussianNB(), 'Probability'),
            'BernoulliNB': (BernoulliNB(), 'Probability'),
            'LDA': (LinearDiscriminantAnalysis(), 'Discriminant'),
            'QDA': (QuadraticDiscriminantAnalysis(), 'Discriminant'),
            'NearestCentroid': (NearestCentroid(), 'Neighbors'),
            
            # Neighbors & SVM
            'KNeighbors': (KNeighborsClassifier(), 'Neighbors'),
            'SVC': (SVC(probability=True), 'SVM'),
            'LinearSVC': (CalibratedClassifierCV(LinearSVC()), 'SVM'),
            
            # Neural Network & Gaussian
            'MLPClassifier': (MLPClassifier(max_iter=500), 'Neural Network'),
            'GaussianProcess': (GaussianProcessClassifier(), 'Gaussian'),
            
            # Baseline
            'DummyClassifier': (DummyClassifier(strategy='most_frequent'), 'Baseline')
        }
        # Evaluation columns: Metrics shifted to Accuracy and F1
        self.columns = ["Model","Group","Train_Acc","Val_Acc","Val_F1","Time_Sec","RAM_GB","File",]
        self.score_df = pd.DataFrame(columns=self.columns)
        self.ignore_list = []
        self.RAM_LIMIT_GB = 10.0
        self.TIME_LIMIT_SEC = 15 * 60  # 900 seconds
        self.methods()

    def methods(self):
        print("set_ignore_list(list)                   : Sets a list of model names to skip")
        print("evaluate(X_train, X_val, y_train, y_val): Runs training loop")
        print("score()                                 : Returns evaluation score")
        print("inspection(model_name_or_file)          : Can Inspect the classification model")
        print("cleanup_models()                        : Delete ALL saved classification models")
        print("zip_models(zip_name)                    : Zip ALL saved classification models\n")
        print("Default RAM_LIMIT_GB and TIME_LIMIT_SEC : 10 GB and 900 seconds")
        
        
    def set_ignore_list(self, models_to_ignore):
        """Sets a list of model names to skip during the marathon."""
        if isinstance(models_to_ignore, list):
            self.ignore_list = models_to_ignore
            print(f"🚫 Ignore list updated: {self.ignore_list}")
        else:
            print("⚠️ Please provide a list of strings.")

    def refresh_score_df(self):
        """Scans directory for .joblib files and updates self.score_df"""
        
        data = []
    
        pattern = re.compile(
            r"(.+)_([0-9\.eE\+\-]+)_([0-9\.eE\+\-]+)_([0-9\.eE\+\-]+)_([0-9\.eE\+\-]+)s_([0-9\.eE\+\-]+)gb\.joblib"
        )
    
        for file in os.listdir('.'):
            match = pattern.search(file)
            if not match:
                continue
    
            name, t_acc, v_acc, v_f1, secs, ram = match.groups()
            group = self.catalog.get(name, (None, "Unknown"))[1]
    
            data.append({
                "Model": name,
                "Group": group,
                "Train_Acc": float(t_acc),
                "Val_Acc": float(v_acc),
                "Val_F1": float(v_f1),
                "Time_Sec": float(secs),
                "RAM_GB": float(ram),
                "File": file,
            })
    
        df = pd.DataFrame(data).reindex(columns=self.columns)
        self.score_df = df.sort_values("Val_Acc", ascending=False).reset_index(drop=True)


    def score(self):
        self.refresh_score_df()
        return self.score_df
        
    def evaluate(self, X_train, X_val, y_train, y_val):
        """Runs training loop with Active Kill for Time and RAM Guarding."""
        self.refresh_score_df()
        print(f"🚀 Classification Marathon Started. Current Progress: {len(self.score_df)} models found.\n")
        
        RAM_LIMIT_GB = self.RAM_LIMIT_GB
        TIME_LIMIT_SEC = self.TIME_LIMIT_SEC
    
        for name, (model, group) in self.catalog.items():
            if name in self.ignore_list:
                print(f"🚫 {name:30} [Ignored by User  ]")
                continue
    
            if name in self.score_df['Model'].values:
                print(f"⏩ {name:30} [Already Evaluated]")
                continue
    
            ram_before = psutil.virtual_memory().used / (1024**3)
            if ram_before > RAM_LIMIT_GB:
                print(f"⚠️ {name:30} [Ignored High RAM ] ({ram_before:.1f}GB)")
                continue
    
            start_time = time.time()
            print(f"⏸️ {name:30} [", end="", flush=True)
    
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            
            p = multiprocessing.Process(target=train_worker, args=(model, X_train, y_train, return_dict))
            p.start()
            p.join(timeout=TIME_LIMIT_SEC)
    
            if p.is_alive():
                p.terminate() 
                p.join()
                print(f" ❌ Ignored High Run Time ]")
                manager.shutdown()
                continue
    
            if not return_dict.get('success', False):
                err = return_dict.get('error', 'Unknown Error')
                print(f" ❌ Fit Failed: {str(err)[:25]} ]")
                manager.shutdown()
                continue
    
            # --- SUCCESS: Retrieve and Score ---
            fitted_model = return_dict['model']
            
            try:
                t_p = fitted_model.predict(X_train)
                t_acc = round(accuracy_score(y_train, t_p), 4)
            
                v_p = fitted_model.predict(X_val)
                v_acc = round(accuracy_score(y_val, v_p), 4)
                v_f1 = round(f1_score(y_val, v_p), 4)
            
                elapsed = round(time.time() - start_time, 1)
            
                ram_after = psutil.virtual_memory().used / (1024**3)
                ram_used = round(ram_after - ram_before, 3)
            
                fname = f"{name}_{t_acc}_{v_acc}_{v_f1}_{elapsed}s_{ram_used}gb.joblib"
                joblib.dump(fitted_model, fname)
                
                # remove references
                del fitted_model
                del t_p, v_p
                return_dict.clear()            
            
                new_row = {
                    "Model": name,
                    "Group": group,
                    "Train_Acc": t_acc,
                    "Val_Acc": v_acc,
                    "Val_F1": v_f1,
                    "Time_Sec": elapsed,
                    "RAM_GB": ram_used,
                    "File": fname
                }
            
                self.score_df = pd.concat(
                    [self.score_df, pd.DataFrame([new_row])],
                    ignore_index=True
                )
            
                print(f" ✅ Completed ] ({elapsed}s, {ram_used}GB )")
            
            except Exception as e:
                print(f" ❌ Scoring Error: {str(e)[:20]} ]")
                
            finally:
                manager.shutdown()
                gc.collect()
                
        self.refresh_score_df()
        
    def inspection(self, model_name_or_file):
        target = model_name_or_file
        if not target.endswith('.joblib'):
            row = self.score_df[self.score_df['Model'] == target]
            if not row.empty:
                target = row.iloc[0]['File']
        
        if os.path.exists(target):
            m = joblib.load(target)
            print(f"\n--- 🔍 Inspecting {target} ---")
            print(f"Class: {type(m).__name__}")
            print(f"Params: {m.get_params()}")
        else:
            print("Model file not found.")

    def cleanup_models(self):
        confirm = input("⚠️ Are you sure you want to delete ALL saved classification models? (y/n): ")
        if confirm.lower() == 'y':
            files_to_remove = [f for f in os.listdir('.') if f.endswith('.joblib')]
            for f in files_to_remove:
                os.remove(f)
            self.refresh_score_df()
            print(f"🧹 Cleaned up {len(files_to_remove)} files.")

    def zip_models(self, zip_name="classification_results"):
        self.refresh_score_df()
        if self.score_df.empty:
            print("⚠️ No models found to zip.")
            return

        zip_filename = f"{zip_name}.zip"
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        
        temp_dir = "temp_cls_zip"
        os.makedirs(temp_dir, exist_ok=True)
        try:
            for file in self.score_df['File']:
                shutil.copy(file, os.path.join(temp_dir, file))
            shutil.make_archive(zip_name, 'zip', temp_dir)
            print(f"📦 Successfully zipped {len(self.score_df)} models at {os.path.abspath(zip_filename)}")
        finally:
            shutil.rmtree(temp_dir)
