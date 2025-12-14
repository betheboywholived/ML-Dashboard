import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Classification Data" / "classification_heart_disease.csv"
TARGET_COL = "target"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_random_forest_heart_disease.pkl"


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data
def prepare_data(df: pd.DataFrame):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


@st.cache_data
def train_and_compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled,
):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1, probability=True, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    use_scaled = {"Logistic Regression", "SVM (RBF)", "KNN"}

    results = []
    fitted_models = {}
    preds = {}
    probs = {}

    for name, model in models.items():
        if name in use_scaled:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te)[:, 1]
        else:
            y_scores = model.decision_function(X_te)
            y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "AUC": auc,
            }
        )

        fitted_models[name] = model
        preds[name] = y_pred
        probs[name] = y_prob

    results_df = pd.DataFrame(results).set_index("Model")
    best_model_name = results_df["F1"].idxmax()
    return results_df, best_model_name, fitted_models, preds, probs


@st.cache_resource
def load_best_model(path: Path):
    if not path.exists():
        return None
    try:
        # Model was saved via joblib.dump in the notebook, so load with joblib.
        model = joblib.load(path)
    except Exception as e:
        # The .pkl is invalid or incompatible; show the reason and disable prediction.
        st.error(f"Failed to load saved model from '{path}': {e}")
        return None
    return model


def plot_metric_bar(results_df: pd.DataFrame, metric: str, key=None):
    df_plot = results_df.reset_index().sort_values(metric, ascending=False)
    fig = px.bar(
        df_plot,
        x="Model",
        y=metric,
        text=df_plot[metric].round(3),
        title=f"Model Comparison – {metric}",
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 1])

    # Use an explicit key to avoid Streamlit duplicate element ID errors
    effective_key = key or f"metric_bar_{metric}"
    st.plotly_chart(fig, use_container_width=True, key=effective_key)


def plot_confusion_matrix(y_true, y_pred, title: str, labels=("0", "1")):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        color_continuous_scale="Blues",
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_roc_curve(y_true, y_prob, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc:.3f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_all_roc_curves(y_true, probs_by_model: dict[str, np.ndarray]):
    """Plot ROC curves for all provided models on a single figure."""
    fig = go.Figure()

    for name, y_prob in probs_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC = {auc:.3f})",
            )
        )

    # Diagonal baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title="ROC Curves for All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef.ravel())
    else:
        st.info("This model does not expose feature importances or coefficients.")
        return

    df_imp = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    fig = px.bar(
        df_imp,
        x="Feature",
        y="Importance",
        title="Feature Importance / Coefficients",
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_imp, use_container_width=True)


def main():
    st.title("Heart Disease Classification – Comparative Analysis")

    # --- DATASET OVERVIEW ---
    st.header("1. Dataset Overview")
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at: {DATA_PATH}")
        return

    df = load_dataset(DATA_PATH)
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Target Distribution")
    st.write(df[TARGET_COL].value_counts())
    st.bar_chart(df[TARGET_COL].value_counts())

    # Correlation heatmap for numeric features
    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu",
            origin="lower",
            title="Feature Correlation Heatmap",
        )
        # Make the heatmap larger for better readability
        fig_corr.update_layout(height=800, width=1000)
        st.plotly_chart(fig_corr, use_container_width=True, key="corr_heatmap")

    # --- DATA PREPARATION ---
    (
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = prepare_data(df)

    # Feature-wise distributions (Plotly equivalents of notebook visuals)
    st.subheader("Feature Distributions")

    # 1) Age distribution
    if "age" in df.columns:
        fig_age = px.histogram(
            df,
            x="age",
            nbins=20,
            title="Age Distribution of Patients",
        )
        fig_age.update_traces(marker_color="skyblue")
        fig_age.update_layout(xaxis_title="Age", yaxis_title="Frequency")
        st.plotly_chart(fig_age, use_container_width=True, key="dist_age")

    # 2) Chest pain type vs target
    if "chest pain type" in df.columns:
        fig_cp = px.histogram(
            df,
            x="chest pain type",
            color=TARGET_COL,
            barmode="group",
            title="Chest Pain Type vs Heart Disease",
        )
        fig_cp.update_layout(
            xaxis_title="Chest Pain Type (1–4)",
            yaxis_title="Count",
            legend_title="Heart Disease",
        )
        st.plotly_chart(fig_cp, use_container_width=True, key="dist_chest_pain")

    # 3) Sex vs target
    if "sex" in df.columns:
        fig_sex = px.histogram(
            df,
            x="sex",
            color=TARGET_COL,
            barmode="group",
            title="Heart Disease by Sex",
        )
        fig_sex.update_layout(
            xaxis_title="Sex (0 = Female, 1 = Male)",
            yaxis_title="Count",
            legend_title="Heart Disease",
        )
        st.plotly_chart(fig_sex, use_container_width=True, key="dist_sex")

    # --- MODEL TRAINING & COMPARISON ---
    st.header("2. Model Training & Comparison")
    with st.spinner("Training models and computing metrics..."):
        results_df, best_model_name, fitted_models, preds, probs = train_and_compare_models(
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_scaled,
            X_test_scaled,
        )

    results_df_sorted = results_df.sort_values("F1", ascending=False)
    results_df_display = results_df_sorted.copy()
    results_df_display["Best"] = results_df_display.index == best_model_name

    st.subheader("Comparison Table")
    st.dataframe(results_df_display.style.format("{:.3f}"), use_container_width=True)
    st.write(f"Best model (by F1-score): **{best_model_name}**")

    # Show full comparison table from previous notebook run if available
    comparison_csv_path = PROJECT_ROOT / "data" / "raw" / "Classification Data" / "comparison_results.csv"
    if comparison_csv_path.exists():
        st.subheader("Full Comparison Table (from notebook)")
        comparison_df = pd.read_csv(comparison_csv_path)

        # Sort by model name so "first of each model" is meaningful
        if "Model" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("Model")

            # Highlight the first row of each model group
            first_indices = (
                comparison_df.reset_index()
                .groupby("Model")["index"]
                .first()
                .values
            )

            def _highlight_first(row):
                if row.name in first_indices:
                    # Light blue background with dark blue text for good contrast
                    style = "background-color: #cce5ff; color: #084298;"
                else:
                    style = ""
                return [style] * len(row)

            styled = comparison_df.style.apply(_highlight_first, axis=1)
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(comparison_df, use_container_width=True)

    st.subheader("Metric Comparison")
    metric = st.selectbox(
        "Select metric to visualize",
        ["Accuracy", "Precision", "Recall", "F1", "AUC"],
        index=3,
    )
    plot_metric_bar(results_df_sorted, metric, key=f"metric_bar_main_{metric}")

    # --- BEST MODEL ANALYSIS ---
    st.header("3. Best Model Analysis")
    best_model = fitted_models[best_model_name]
    y_pred_best = preds[best_model_name]
    y_prob_best = probs[best_model_name]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix (Best Model)")
        plot_confusion_matrix(
            y_test,
            y_pred_best,
            title=f"Confusion Matrix – {best_model_name}",
            labels=("0", "1"),
        )
    with col2:
        st.subheader("ROC Curve (Best Model)")
        plot_roc_curve(
            y_test,
            y_prob_best,
            title=f"ROC Curve – {best_model_name}",
        )

    st.subheader("Feature Importance / Coefficients")
    plot_feature_importance(best_model, X.columns.tolist())

    st.subheader("ROC Curves for All Models")
    plot_all_roc_curves(y_test, probs)

    # --- PREDICTION SECTION (SAVED BEST MODEL) ---
    st.header("4. Prediction with Saved Best Model")

    saved_model = load_best_model(BEST_MODEL_PATH)
    if saved_model is None:
        st.warning(
            f"Saved best model could not be loaded from `{BEST_MODEL_PATH}`. "
            "Re-export the model from the notebook as a valid pickle (e.g. via pickle.dump "
            "or joblib.dump) and ensure this path points to it."
        )
        return

    # Batch prediction from CSV
    st.subheader("4.1 Batch Prediction from CSV")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the same feature columns.",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if TARGET_COL in batch_df.columns:
            batch_features = batch_df.drop(columns=[TARGET_COL])
        else:
            batch_features = batch_df

        # If your saved model is a pipeline, this will handle preprocessing internally.
        # Otherwise, adapt to apply the same preprocessing as above before prediction.
        try:
            batch_pred = saved_model.predict(batch_features)
            if hasattr(saved_model, "predict_proba"):
                batch_prob = saved_model.predict_proba(batch_features)[:, 1]
            else:
                batch_prob = None
        except Exception as e:
            st.error(f"Error during prediction: {e}")
        else:
            result_df = batch_df.copy()
            result_df["prediction"] = batch_pred
            if batch_prob is not None:
                result_df["probability_positive"] = batch_prob

            st.dataframe(result_df.head(), use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )

    # Single-row prediction
    st.subheader("4.2 Single Input Prediction")

    with st.form("single_prediction_form"):
        inputs = {}
        for col in X.columns:
            col_dtype = X[col].dtype
            col_min = X[col].min()
            col_max = X[col].max()

            if np.issubdtype(col_dtype, np.integer):  # type: ignore
                default_val = int(X[col].median())
                min_val = int(col_min)
                max_val = int(col_max)
                inputs[col] = st.number_input(
                    col,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=1,
                )
            else:
                default_val = float(X[col].median())
                min_val = float(col_min)
                max_val = float(col_max)
                inputs[col] = st.number_input(
                    col,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    format="%.3f",
                )

        submitted = st.form_submit_button("Predict")

    if submitted:
        single_df = pd.DataFrame([inputs])

        try:
            single_pred = saved_model.predict(single_df)[0]
            if hasattr(saved_model, "predict_proba"):
                single_prob = saved_model.predict_proba(single_df)[0, 1]
            else:
                single_prob = None
        except Exception as e:
            st.error(f"Error during prediction: {e}")
        else:
            st.write(f"Predicted class: **{int(single_pred)}**")
            if single_prob is not None:
                st.write(
                    f"Predicted probability of positive class (1): **{single_prob:.3f}**"
                )


# Streamlit page entry point
main()
