## ML Dashboard – Heart Disease Classification

An interactive Streamlit dashboard for experimenting with machine learning models.

The main implemented use case is a heart disease **classification** experiment that
recreates the analysis from the Jupyter notebook in `notebooks/ML2_Heart_Disease.ipynb`.
A simple **regression** placeholder page is also available.

---

## Project structure

Key folders and files:

- `app/app.py` – main Streamlit entry point (overview page).
- `app/pages/classification.py` – heart disease classification dashboard.
- `app/pages/regression.py` – regression experiments (currently minimal).
- `notebooks/ML2_Heart_Disease.ipynb` – original notebook used to explore and train models.
- `data/raw/Classification Data/classification_heart_disease.csv` – heart disease dataset.
- `data/raw/Classification Data/comparison_results.csv` – comparison table exported from the notebook.
- `models/best_random_forest_heart_disease.pkl` – saved best classification model (Random Forest).

Run all commands from the project root (where this `Readme.md` and `requirements.txt` live).

---

## Data & model requirements

To use the classification page as intended you need:

1. **Dataset**
	 - Place the heart disease CSV at:
		 - `data/raw/Classification Data/classification_heart_disease.csv`

2. **Comparison results (optional but recommended)**
	 - If present, the page will show an additional table of results exported from the notebook:
		 - `data/raw/Classification Data/comparison_results.csv`

3. **Saved best model for prediction**
	 - The prediction section uses a pre-trained Random Forest model saved with `joblib` at:
		 - `models/best_random_forest_heart_disease.pkl`
	 - You can recreate this file by running the final cells of
		 `notebooks/ML2_Heart_Disease.ipynb`, which train the tuned Random Forest
		 and save it to the `models/` folder.

If the model file is missing, the classification page will still load and you can
train and compare models interactively, but the prediction section will not work
until the file is created.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure you are using a Python version compatible with the packages in `requirements.txt`
(e.g. Python 3.9+).

---

## Run the app

From the project root, start Streamlit with:

```bash
streamlit run app/app.py
```

Streamlit will open the app in your browser. Use the left-hand sidebar to switch
between the **Classification** and **Regression** pages.

---

## Classification page overview

The classification page (`app/pages/classification.py`) is organized into several
sections that mirror the notebook workflow:

1. **Dataset overview**
	 - Loads the heart disease dataset and shows basic info, a target distribution,
		 a correlation heatmap, and key feature distributions.

2. **Model training & comparison**
	 - Splits the data into train/test sets with a fixed random state.
	 - Trains multiple classifiers (Logistic Regression, KNN, SVM, Random Forest,
		 Decision Tree, Gradient Boosting).
	 - Computes metrics such as Accuracy, Precision, Recall, F1-score, and AUC.
	 - Displays a sortable metrics table and interactive bar charts.

3. **Full comparison table (from notebook)**
	 - If `comparison_results.csv` is available, shows a detailed comparison of
		 many model configurations, with per-model best rows highlighted.

4. **Best model analysis**
	 - Focuses on the best-performing model (by F1-score) on the test set.
	 - Shows confusion matrix, ROC curve, feature importances/coefficients,
		 and ROC curves for all trained models.

5. **Prediction**
	 - Uses the saved best Random Forest model from `models/best_random_forest_heart_disease.pkl`.
	 - Supports batch prediction from an uploaded CSV.
	 - Provides a single-input form where each feature is constrained to the
		 min/max values observed in the dataset.

---

## Regression page

The regression page (`app/pages/regression.py`) currently contains a basic
placeholder layout. You can extend it with your own regression experiments
following a similar pattern to the classification page.

