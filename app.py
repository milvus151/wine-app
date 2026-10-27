import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Подбор вина", page_icon="🍷")
st.title("Подбор вина")

@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")

@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / "test.csv")

@st.cache_data
def get_test_probabilities(_model, X):
    return _model.predict_proba(X)[:, 1]

model = load_model()
config = json.loads(
    (APP_DIR / "config.json").read_text(encoding="utf-8")
)
test_data = load_test_data()
feature_columns = [name for name in test_data.columns if name != "target"]

threshold = st.sidebar.slider(
    "Порог для рекомендации",
    min_value=0.0,
    max_value=1.0,
    value=float(config["threshold"]),
    step=0.0001,
    format="%.4f",
)

single_wine_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs(
    [
        "Оценка одного вина",
        "Модельные метрики",
        "Пакетное предсказание",
        "Что выучила модель",
    ]
)

with single_wine_tab:
    st.write("Введите характеристики вина.")
    wine_values = {}
    input_columns = st.columns(3)
    numeric_features = [
        name for name in feature_columns if name != "is_red"
    ]
    for index, name in enumerate(numeric_features):
        values = test_data[name]
        decimals = 4 if name == "density" else 3 if name == "chlorides" else 2
        with input_columns[index % 3]:
            wine_values[name] = st.number_input(
                name.replace("_", " ").capitalize(),
                min_value=float(values.min()),
                max_value=float(values.max()),
                value=float(values.median()),
                step=10 ** (-decimals),
                format=f"%.{decimals}f",
                key=name,
            )
    is_red = st.toggle("Красное вино", value=False)
    st.caption("Переключатель выключен - белое вино.")
    wine_values["is_red"] = int(is_red)
    wine = pd.DataFrame([wine_values], columns=feature_columns)
    probability = float(model.predict_proba(wine)[0, 1])
    st.metric("Вероятность хорошего вина", f"{probability:.1%}")
    if probability >= threshold:
        st.success("Модель рекомендует добавить это вино в подборку.")
    else:
        st.warning("Модель не рекомендует добавлять это вино в подборку.")

with metrics_tab:
    st.subheader("Качество модели на тестовой выборке")
    y_true = test_data["target"].to_numpy()
    test_features = test_data[feature_columns]
    probabilities = get_test_probabilities(model, test_features)
    y_pred = (probabilities >= threshold).astype(int)
    confusion = metrics.confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )
    tn, fp, fn, tp = confusion.ravel()
    precision = metrics.precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = metrics.recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = metrics.f1_score(
        y_true, y_pred, zero_division=0
    )
    col_precision, col_recall, col_f1 = st.columns(3)
    col_precision.metric("Precision", f"{precision:.3f}")
    col_recall.metric("Recall", f"{recall:.3f}")
    col_f1.metric("F_1", f"{f1:.3f}")
    st.write("Матрица ошибок")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["Класс 0", "Класс 1"],
    ).plot(ax=ax_cm, cmap="Blues", colorbar=False)
    st.pyplot(fig_cm)
    plt.close(fig_cm)
    fpr_curve, tpr_curve, _ = metrics.roc_curve(
        y_true, probabilities
    )
    precision_curve, recall_curve, _ = (
        metrics.precision_recall_curve(y_true, probabilities)
    )
    roc_auc = metrics.roc_auc_score(y_true, probabilities)
    pr_auc = metrics.average_precision_score(
        y_true, probabilities
    )
    current_fpr = fp / (fp + tn)
    current_tpr = recall
    current_pr_precision = precision if tp + fp > 0 else 1.0
    fig_curves, (ax_roc, ax_pr) = plt.subplots(
        1, 2, figsize=(12, 5)
    )
    ax_roc.plot(
        fpr_curve, tpr_curve,
        label=f"ROC-AUC = {roc_auc:.3f}",
    )
    ax_roc.scatter(
        current_fpr, current_tpr,
        s=90, color="red", zorder=3,
        label=f"Порог {threshold:.4f}",
    )
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set(
        title="ROC-кривая",
        xlabel="False Positive Rate",
        ylabel="Recall",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax_roc.legend()
    ax_roc.grid()
    ax_pr.plot(
        recall_curve, precision_curve,
        label=f"PR-AUC (AP) = {pr_auc:.3f}",
    )
    ax_pr.scatter(
        recall, current_pr_precision,
        s=90, color="red", zorder=3,
        label=f"Порог {threshold:.4f}",
    )
    ax_pr.axhline(
        y_true.mean(), linestyle="--", color="gray",
        label="Доля хороших вин",
    )
    ax_pr.set(
        title="PR-кривая",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax_pr.legend()
    ax_pr.grid()
    fig_curves.tight_layout()
    st.pyplot(fig_curves)
    plt.close(fig_curves)
    st.info(
        "Если увеличить порог, в подборку попадёт меньше вин: "
        "посредственных рекомендаций станет меньше, но часть хороших вин "
        "будет пропущена. Если уменьшить порог, хороших вин найдётся больше, "
        "но возрастёт число посредственных рекомендаций."
    )

with batch_tab:
    st.subheader("Оценка нескольких вин")
    st.write(
        "Загрузите CSV-файл с 12 признаками вина. "
        "Столбец target не нужен."
    )
    uploaded_file = st.file_uploader(
        "CSV-файл с винами",
        type="csv",
    )
    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            st.error("Не удалось прочитать CSV-файл. Проверьте его содержимое. ")
        else:
            missing_columns = [
                name for name in feature_columns
                if name not in uploaded_data.columns
            ]
            if missing_columns:
                st.error(
                    "В файле не хватает столбцов: "
                    + ", ".join(missing_columns)
                )
            elif uploaded_data.empty:
                st.error("CSV-файл не содержит вин.")
            else:
                uploaded_features = uploaded_data[feature_columns].copy()
                for name in feature_columns:
                    uploaded_features[name] = pd.to_numeric(
                        uploaded_features[name],
                        errors="coerce",
                    )
                if uploaded_features.isna().any().any():
                    st.error(
                        "В признаках есть пустые или нечисловые значения. "
                        "Проверьте CSV-файл."
                    )
                elif not uploaded_features["is_red"].isin([0, 1]).all():
                    st.error("В столбце is_red допустимы только 0 и 1.")
                else:
                    probabilities = model.predict_proba(
                        uploaded_features
                    )[:, 1]
                    result = uploaded_data.copy()
                    result["Вероятность хорошего вина"] = probabilities
                    result["Решение"] = [
                        "Рекомендовать"
                        if probability >= threshold
                        else "Не рекомендовать"
                        for probability in probabilities
                    ]
                    st.write(f"Текущий порог: {threshold:.4f}")
                    st.dataframe(result)
                    st.download_button(
                        "Скачать результаты в CSV",
                        data=result.to_csv(index=False).encode("utf-8-sig"),
                        file_name="wine_predictions.csv",
                        mime="text/csv",
                    )

with coefficients_tab:
    st.subheader("Коэффициенты логистической регрессии")
    logistic_regression = model.named_steps["logisticregression"]
    coefficients = pd.Series(
        logistic_regression.coef_[0],
        index=model.feature_names_in_,
    ).sort_values()
    colors = [
        "red" if value < 0 else "green"
        for value in coefficients
    ]
    fig_coef, ax_coef = plt.subplots(figsize=(9, 6))
    ax_coef.barh(coefficients.index, coefficients.values, color=colors)
    ax_coef.axvline(0, color="black", linewidth=1)
    ax_coef.set(
        xlabel="Коэффициент после масштабирования",
        ylabel="Признак",
        title="Связь признаков с хорошим вином",
    )
    fig_coef.tight_layout()
    st.pyplot(fig_coef)
    plt.close(fig_coef)
    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)
    if not positive.empty:
        st.write(
            "Сильнее всего за хорошее вино: "
            + ", ".join(positive.index)
            + "."
        )
    if not negative.empty:
        st.write(
            "Сильнее всего против хорошего вина: "
            + ", ".join(negative.index)
            + "."
        )
    st.caption(
        "Зелёные столбцы - положительные коэффициенты, "
        "красные - отрицательные. Коэффициенты относятся к признакам "
        "после StandardScaler; модель здесь не переобучается."
    )