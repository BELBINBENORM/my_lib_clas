from setuptools import setup

setup(
    name="select_classification",
    version="0.1",
    py_modules=["select_classification"], 
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib',
        'psutil',      
        'matplotlib',
        'xgboost',
        'lightgbm',
        'catboost',
    ],
    author="Belbin Beno R M",
    description="An automated marathon evaluation tool for classification models.",
)
