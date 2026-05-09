import pandas as pd
import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              BaggingRegressor, RandomForestRegressor,
                              AdaBoostRegressor, GradientBoostingRegressor)
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(layout="wide")
st.title('Machine Learning Model')

img_path = os.path.join(os.getcwd(), "static", "Machine learning.jpg")
if os.path.exists(img_path):
    img = Image.open(img_path)
    img = img.resize((1500, 600))
    st.image(img)

tab1, tab2, tab3, tab4 = st.tabs(['Tab1', 'Tab2', 'Tab3', 'Tab4'])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    Problem_type  = st.selectbox("Problem Type", ['Classification', 'Regression'])
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)

        Target = st.selectbox('Target', df.columns)

        st.subheader('Top 5 rows')
        st.dataframe(df.head())

        st.subheader('Bottom 5 rows')
        st.dataframe(df.tail())
        st.write(f"This dataset includes {df.shape[1]} columns and {df.shape[0]} rows")

        st.subheader('Statistical Summary')
        st.dataframe(df.describe(include="all"))

        st.subheader("Dataset Info")
        info_df = pd.DataFrame({
            "Column":        df.columns,
            "Data Type":     df.dtypes.values,
            "Non-Null Count":df.notnull().sum().values,
            "Null Count":    df.isnull().sum().values
        })
        st.dataframe(info_df)

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = categorical_cols[df[categorical_cols].nunique() <= 10]
        numerical_cols   = df.select_dtypes(include=['number']).columns
        st.write(f"This dataset has {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")

        st.subheader('Duplicate Values')
        duplicates = df.duplicated().sum()
        if duplicates == 0:
            st.success("No duplicates found.")
        else:
            st.warning(f"{duplicates} duplicate rows found.")

        st.subheader("Null Values")
        nulls = df.isnull().sum()
        st.write(nulls[nulls > 0] if nulls.sum() > 0 else "No null values found.")

        # ✅ Value counts BEFORE saving to session state — runs immediately
        st.subheader('Value Counts')
        col_vc = st.selectbox('Select a column to see value counts:', df.columns)
        st.write(df[col_vc].value_counts())

        st.session_state['df']               = df
        st.session_state['categorical_cols'] = categorical_cols
        st.session_state['numerical_cols']   = numerical_cols
        st.session_state['Target']           = Target
        st.session_state['Problem_type']     = Problem_type

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Visualisation
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if 'df' not in st.session_state:
        st.warning("Please upload a dataset in Tab 1 first.")
        st.stop()

    df               = st.session_state['df']
    categorical_cols = st.session_state['categorical_cols']
    numerical_cols   = st.session_state['numerical_cols']
    Target           = st.session_state['Target']
    Problem_type     = st.session_state['Problem_type']

    # ── Univariate ────────────────────────────────────────────────────────────
    st.header("Univariate Analysis")

    if len(categorical_cols) > 0:
        st.subheader("Categorical Columns")
        for feat in categorical_cols:
            total = len(df[feat])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            sns.countplot(data=df, x=feat, palette='viridis',
                          order=df[feat].value_counts().index, ax=ax1)
            for p in ax1.patches:
                ax1.annotate("{:.1f}%".format(p.get_height() / total * 100),
                             (p.get_x() + p.get_width() / 2, p.get_height()),
                             ha='center', va='bottom', fontsize=8, fontweight='bold',
                             xytext=(0, 3), textcoords='offset points')
            ax1.set_title(feat, fontweight='bold', color='red')
            ax1.tick_params(axis='x', rotation=45)
            sns.boxplot(data=df, x=feat, palette='viridis', ax=ax2)
            ax2.set_title(f"Boxplot of {feat}", fontweight='bold', color='red')
            ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if len(numerical_cols) > 0:
        st.subheader("Numerical Columns")
        for feat in numerical_cols:
            fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(14, 4))
            sns.histplot(data=df, x=feat, kde=True, ax=ax_hist)
            ax_hist.axvline(df[feat].mean(),   color='red',  linestyle='--', label='Mean')
            ax_hist.axvline(df[feat].median(), color='blue', linestyle='--', label='Median')
            ax_hist.legend()
            ax_hist.set_title('Distribution of ' + feat, color='red', fontweight='bold')
            sns.boxplot(data=df, x=feat, ax=ax_box)
            ax_box.set_title('Boxplot of ' + feat, color='red', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Bivariate ─────────────────────────────────────────────────────────────
    st.header("Bivariate Analysis")

    if Problem_type == 'Classification':
        # Categorical vs Target
        if len(categorical_cols) > 0:
            st.subheader("Categorical Columns vs Target")
            for feat in categorical_cols:
                if feat == Target:
                    continue
                fig, ax = plt.subplots(figsize=(14, 4))
                sns.countplot(data=df, x=feat, hue=df[Target],
                              order=df[feat].value_counts().index, palette='Set2', ax=ax)
                for container in ax.containers:
                    ax.bar_label(container)
                ax.set_title(f"{feat} vs {Target}", fontweight='bold', color='red')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Numerical vs Target
        if len(numerical_cols) > 0:
            st.subheader("Numerical Columns vs Target")
            for feat in numerical_cols:
                if feat == Target:
                    continue
                target_uniq = df[Target].unique()
                if len(target_uniq) < 2:
                    continue
                fig, axs = plt.subplots(1, 2, figsize=(14, 4))
                sns.boxplot(data=df, x=Target, y=feat, ax=axs[0], palette="Set2")
                axs[0].set_title(f"Boxplot: {feat} by {Target}", fontweight='bold', color='red')
                sns.boxplot(data=df, x=Target, y=feat, ax=axs[1],
                            showfliers=False, palette="Set2")
                axs[1].set_title(f"Boxplot (no outliers): {feat} by {Target}",
                                 fontweight='bold', color='red')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    else:
        # ✅ Regression: plot numerical vs numerical and categorical vs numerical
        # Numerical vs Numerical (scatter with regression line)
        non_target_num = [c for c in numerical_cols if c != Target]
        if len(non_target_num) > 0:
            st.subheader("Numerical Features vs Target")
            for feat in non_target_num:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(df[feat], df[Target], alpha=0.4, color='steelblue', edgecolors='k',
                           linewidths=0.2, s=20)
                m, b = np.polyfit(df[feat].dropna(), df[Target][df[feat].notna()], 1)
                x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
                ax.plot(x_line, m * x_line + b, color='red', lw=2, label='Trend')
                ax.set_xlabel(feat)
                ax.set_ylabel(Target)
                ax.set_title(f"{feat} vs {Target}", fontweight='bold', color='red')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Categorical vs Target (boxplot — much faster than countplot with continuous target)
        if len(categorical_cols) > 0:
            st.subheader("Categorical Features vs Target")
            for feat in categorical_cols:
                if feat == Target:
                    continue
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                sns.boxplot(data=df, x=feat, y=Target, palette='Set2', ax=ax1)
                ax1.set_title(f"{feat} vs {Target}", fontweight='bold', color='red')
                ax1.tick_params(axis='x', rotation=45)
                sns.boxplot(data=df, x=feat, y=Target, palette='Set2',
                            showfliers=False, ax=ax2)
                ax2.set_title(f"{feat} vs {Target} (no outliers)", fontweight='bold', color='red')
                ax2.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Multivariate ──────────────────────────────────────────────────────────
    st.header("Multivariate Analysis")

    if len(numerical_cols) > 1:
        corr = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap", fontweight='bold', fontsize=16)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.boxplot(data=df[numerical_cols], ax=ax)
        ax.set_title('Boxplots — Numerical Features', fontweight='bold', fontsize=16)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Pairplot only if small enough (avoid huge memory usage)
    selected_cols = [c for c in numerical_cols if c != Target][:4] + [Target]
    selected_cols = list(dict.fromkeys(selected_cols))  # deduplicate
    if len(selected_cols) >= 2:
        if Problem_type == 'Classification':
            g = sns.pairplot(df[selected_cols], hue=Target, plot_kws={'alpha': 0.4})
        else:
            g = sns.pairplot(df[selected_cols], plot_kws={'alpha': 0.4})
        g.fig.suptitle("Pairwise Scatterplots", y=1.02, fontweight='bold')
        st.pyplot(g.fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if 'df' not in st.session_state:
        st.warning("Please upload a dataset in Tab 1 first.")
        st.stop()

    df               = st.session_state['df']
    categorical_cols = st.session_state['categorical_cols']
    numerical_cols   = st.session_state['numerical_cols']
    Target           = st.session_state['Target']
    Problem_type     = st.session_state['Problem_type']

    x = df.drop(Target, axis=1)
    y = df[Target]

    @st.fragment
    def split_section(x, y):
        test_size         = st.slider("Select the % of test data", min_value=20, max_value=50)
        test_size_decimal = test_size / 100
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, stratify=y, test_size=test_size_decimal, random_state=1)
            x_val, x_test, y_val, y_test = train_test_split(
                x_test, y_test, stratify=y_test, test_size=0.50, random_state=1)
        except ValueError:
            st.warning("Stratified split failed — using regular split instead.")
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size_decimal, random_state=1)
            x_val, x_test, y_val, y_test = train_test_split(
                x_test, y_test, test_size=0.50, random_state=1)

        st.write(f"Training size: {x_train.shape[0]} rows")
        st.write(f"Validation size: {x_val.shape[0]} rows")
        st.write(f"Test size: {x_test.shape[0]} rows")

        st.session_state['x_train'] = x_train
        st.session_state['x_test']  = x_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test']  = y_test
        st.session_state['x_val']   = x_val
        st.session_state['y_val']   = y_val

    split_section(x, y)

    if 'x_train' in st.session_state:
        x_train = st.session_state['x_train']
        x_val   = st.session_state['x_val']
        x_test  = st.session_state['x_test']

        cat_cols = [c for c in categorical_cols if c in x_train.columns and c != Target]
        num_cols = [c for c in numerical_cols   if c in x_train.columns and c != Target]

        x_train_cat = pd.get_dummies(x_train[cat_cols], drop_first=True)
        x_val_cat   = pd.get_dummies(x_val[cat_cols],   drop_first=True)
        x_test_cat  = pd.get_dummies(x_test[cat_cols],  drop_first=True)
        x_val_cat   = x_val_cat.reindex(columns=x_train_cat.columns,  fill_value=0)
        x_test_cat  = x_test_cat.reindex(columns=x_train_cat.columns, fill_value=0)

        scaler      = StandardScaler()
        x_train_num = pd.DataFrame(scaler.fit_transform(x_train[num_cols]),
                                   columns=num_cols, index=x_train.index)
        x_val_num   = pd.DataFrame(scaler.transform(x_val[num_cols]),
                                   columns=num_cols, index=x_val.index)
        x_test_num  = pd.DataFrame(scaler.transform(x_test[num_cols]),
                                   columns=num_cols, index=x_test.index)

        x_train_final = pd.concat([x_train_num, x_train_cat], axis=1)
        x_val_final   = pd.concat([x_val_num,   x_val_cat],   axis=1)
        x_test_final  = pd.concat([x_test_num,  x_test_cat],  axis=1)

        st.success("Data preprocessed successfully!")
        st.dataframe(x_train_final.head())

        if Problem_type == 'Classification':
            from sklearn.preprocessing import LabelEncoder
            le      = LabelEncoder()
            y_train = le.fit_transform(st.session_state['y_train'])
            y_val   = le.transform(st.session_state['y_val'])
            y_test  = le.transform(st.session_state['y_test'])
        else:
            y_train = st.session_state['y_train']
            y_val   = st.session_state['y_val']
            y_test  = st.session_state['y_test']

        x_train_os = x_train_final
        y_train_os = y_train

        if Problem_type == 'Classification':
            st.subheader("Class Balance Check")
            balance = pd.Series(y_train).value_counts(normalize=True) * 100
            st.write(balance)
            if balance.min() < 20:
                st.warning(f"Dataset is imbalanced! Minority class: {balance.min():.2f}%")
                if st.checkbox("Apply SMOTE to balance the data?"):
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=1)
                    x_train_os, y_train_os = smote.fit_resample(x_train_final, y_train)
                    st.success(f"SMOTE applied! New training size: {x_train_os.shape[0]} rows")
                    st.write(pd.Series(y_train_os).value_counts(normalize=True) * 100)
            else:
                st.success(f"Dataset is balanced! Minority class: {balance.min():.2f}%")

        st.session_state['x_train_final'] = x_train_final
        st.session_state['x_val_final']   = x_val_final
        st.session_state['x_test_final']  = x_test_final
        st.session_state['y_train']       = y_train
        st.session_state['y_val']         = y_val
        st.session_state['y_test']        = y_test
        st.session_state['x_train_os']    = x_train_os
        st.session_state['y_train_os']    = y_train_os

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Modelling
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if 'x_train_final' not in st.session_state:
        st.warning("Please complete preprocessing in Tab 3 first.")
        st.stop()

    x_train_final = st.session_state['x_train_final']
    x_train_os    = st.session_state['x_train_os']
    x_val_final   = st.session_state['x_val_final']
    x_test_final  = st.session_state['x_test_final']
    y_train       = st.session_state['y_train']
    y_train_os    = st.session_state['y_train_os']
    y_val         = st.session_state['y_val']
    y_test        = st.session_state['y_test']
    Problem_type  = st.session_state['Problem_type']

    # ══════════════════════════════════════════════════════════════════════════
    # CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    if Problem_type == 'Classification':

        def model_performance_classification(model, predictors, target, threshold=0.5):
            n_classes  = len(np.unique(target))
            avg        = 'binary' if n_classes == 2 else 'weighted'
            if n_classes == 2:
                prob_pred  = model.predict_proba(predictors)[:, 1]
                class_pred = [1 if i >= threshold else 0 for i in prob_pred]
            else:
                class_pred = model.predict(predictors)
            return pd.DataFrame({
                "Accuracy":  [round(accuracy_score(target,  class_pred), 4)],
                "Recall":    [round(recall_score(target,    class_pred, average=avg), 4)],
                "Precision": [round(precision_score(target, class_pred, average=avg), 4)],
                "F1":        [round(f1_score(target,        class_pred, average=avg), 4)],
            })

        def model_performances(models, x, y, xval, yval):
            n_classes = len(np.unique(y))
            st.subheader("Train vs Validation Performance")
            for i in range(0, len(models), 2):
                pair = models[i:i+2]
                cols = st.columns(len(pair))
                for col, (name, model) in zip(cols, pair):
                    model.fit(x, y)
                    with col:
                        st.markdown(f"**{name}**")
                        st.markdown("🟦 Train")
                        st.dataframe(model_performance_classification(model, x, y),
                                     use_container_width=True)
                        st.markdown("🟧 Validation")
                        st.dataframe(model_performance_classification(model, xval, yval),
                                     use_container_width=True)

            st.subheader("Cross Validation (Mean Scores)")
            skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
            scoring = (['accuracy', 'precision', 'recall', 'f1'] if n_classes == 2
                       else ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
            for i in range(0, len(models), 2):
                pair = models[i:i+2]
                cols = st.columns(len(pair))
                for col, (name, model) in zip(cols, pair):
                    scores = cross_validate(model, x, y, scoring=scoring,
                                            cv=skf, return_train_score=False)
                    w = "_weighted" if n_classes > 2 else ""
                    with col:
                        st.markdown(f"**{name}** (CV)")
                        st.dataframe(pd.DataFrame({
                            'Accuracy':  [round(scores['test_accuracy'].mean(), 4)],
                            'Recall':    [round(scores[f'test_recall{w}'].mean(), 4)],
                            'Precision': [round(scores[f'test_precision{w}'].mean(), 4)],
                            'F1':        [round(scores[f'test_f1{w}'].mean(), 4)],
                        }), use_container_width=True)

        def AUC_ROC(models, x, y, xval, yval):
            st.subheader("AUC-ROC Scores")
            n_classes = len(np.unique(y))
            rows = []
            for name, model in models:
                model.fit(x, y)
                try:
                    if n_classes == 2:
                        auc_t = roc_auc_score(y,    model.predict_proba(x)[:, 1])
                        auc_v = roc_auc_score(yval, model.predict_proba(xval)[:, 1])
                    else:
                        auc_t = roc_auc_score(y,    model.predict_proba(x),
                                              multi_class='ovr', average='weighted')
                        auc_v = roc_auc_score(yval, model.predict_proba(xval),
                                              multi_class='ovr', average='weighted')
                    rows.append({"Model": name, "Train AUC": round(auc_t, 4),
                                 "Val AUC": round(auc_v, 4)})
                except Exception as e:
                    rows.append({"Model": name, "Train AUC": "N/A", "Val AUC": "N/A"})
                    st.warning(f"{name} AUC-ROC skipped: {e}")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        def score_dif(models, x, y, xval, yval):
            st.subheader("Train vs Validation Differences")
            n_classes = len(np.unique(y))
            avg       = 'binary' if n_classes == 2 else 'weighted'
            for i in range(0, len(models), 2):
                pair = models[i:i+2]
                cols = st.columns(len(pair))
                for col, (name, model) in zip(cols, pair):
                    model.fit(x, y)
                    ytr = model.predict(x)
                    yvl = model.predict(xval)
                    d   = pd.DataFrame({
                        "Metric": ["Accuracy", "Recall", "Precision", "F1"],
                        "Train":  [round(accuracy_score(y,   ytr), 4),
                                   round(recall_score(y,     ytr, average=avg), 4),
                                   round(precision_score(y,  ytr, average=avg), 4),
                                   round(f1_score(y,         ytr, average=avg), 4)],
                        "Val":    [round(accuracy_score(xval if False else yval,  yvl), 4),
                                   round(recall_score(yval,    yvl, average=avg), 4),
                                   round(precision_score(yval, yvl, average=avg), 4),
                                   round(f1_score(yval,        yvl, average=avg), 4)],
                    })
                    d["Diff"] = (d["Train"] - d["Val"]).round(4)
                    with col:
                        st.markdown(f"**{name}**")
                        st.dataframe(d, use_container_width=True)

        par_models = [
            ("Logistic Regression", LogisticRegression()),
            ("Naive Bayes",         GaussianNB()),
        ]
        non_para_models = [
            ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=3)),
            ("Decision Tree",       DecisionTreeClassifier(random_state=1)),
            ("Bagging",             BaggingClassifier(
                                        estimator=DecisionTreeClassifier(random_state=1),
                                        random_state=1)),
            ("Random Forest",       RandomForestClassifier(random_state=1)),
            ("AdaBoost",            AdaBoostClassifier(random_state=1)),
            ("GBM",                 GradientBoostingClassifier(random_state=1)),
            ("XGBoost",             XGBClassifier(random_state=1, eval_metric='logloss')),
            ("SVM",                 SVC(kernel='rbf', C=1.0, gamma='scale',
                                        random_state=1, probability=True)),
        ]

        st.header("Parametric Models")
        st.subheader("Original Data")
        model_performances(par_models, x_train_final, y_train, x_val_final, y_val)
        AUC_ROC(par_models, x_train_final, y_train, x_val_final, y_val)
        score_dif(par_models, x_train_final, y_train, x_val_final, y_val)
        st.subheader("Oversampled Data")
        model_performances(par_models, x_train_os, y_train_os, x_val_final, y_val)
        AUC_ROC(par_models, x_train_os, y_train_os, x_val_final, y_val)
        score_dif(par_models, x_train_os, y_train_os, x_val_final, y_val)

        st.header("Non-Parametric Models")
        st.subheader("Original Data")
        model_performances(non_para_models, x_train_final, y_train, x_val_final, y_val)
        AUC_ROC(non_para_models, x_train_final, y_train, x_val_final, y_val)
        score_dif(non_para_models, x_train_final, y_train, x_val_final, y_val)
        st.subheader("Oversampled Data")
        model_performances(non_para_models, x_train_os, y_train_os, x_val_final, y_val)
        AUC_ROC(non_para_models, x_train_os, y_train_os, x_val_final, y_val)
        score_dif(non_para_models, x_train_os, y_train_os, x_val_final, y_val)

        st.header("Best Model Selection")
        from sklearn.model_selection import RandomizedSearchCV
        avg = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'

        results = []
        for name, model in par_models + non_para_models:
            model.fit(x_train_final, y_train)
            ytp = model.predict(x_train_final)
            yvp = model.predict(x_val_final)
            results.append({
                "Model":           name,
                "Train Accuracy":  round(accuracy_score(y_train,  ytp), 4),
                "Train F1":        round(f1_score(y_train,        ytp, average=avg), 4),
                "Train Recall":    round(recall_score(y_train,    ytp, average=avg), 4),
                "Train Precision": round(precision_score(y_train, ytp, average=avg), 4),
                "Val Accuracy":    round(accuracy_score(y_val,    yvp), 4),
                "Val F1":          round(f1_score(y_val,          yvp, average=avg), 4),
                "Val Recall":      round(recall_score(y_val,      yvp, average=avg), 4),
                "Val Precision":   round(precision_score(y_val,   yvp, average=avg), 4),
            })
        results_df = pd.DataFrame(results).sort_values("Val F1", ascending=False)

        st.subheader("Model Comparison (Before Tuning)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("🟦 **Train Scores**")
            st.dataframe(results_df[["Model","Train Accuracy","Train F1",
                                     "Train Recall","Train Precision"]].reset_index(drop=True),
                         use_container_width=True)
        with c2:
            st.markdown("🟧 **Validation Scores**")
            st.dataframe(results_df[["Model","Val Accuracy","Val F1",
                                     "Val Recall","Val Precision"]].reset_index(drop=True),
                         use_container_width=True)

        param_grids = {
            "Logistic Regression": (LogisticRegression(max_iter=1000),
                {"C": [0.01,0.1,1,10,100], "solver": ["lbfgs","liblinear"]}),
            "Naive Bayes": (GaussianNB(),
                {"var_smoothing": np.logspace(-12,-6,10)}),
            "K-Nearest Neighbors": (KNeighborsClassifier(),
                {"n_neighbors":[3,5,7,9,11],"weights":["uniform","distance"],
                 "metric":["euclidean","manhattan"]}),
            "Decision Tree": (DecisionTreeClassifier(random_state=1),
                {"max_depth":[3,5,7,10,None],"min_samples_split":[2,5,10],
                 "criterion":["gini","entropy"]}),
            "Bagging": (BaggingClassifier(estimator=DecisionTreeClassifier(random_state=1),
                                          random_state=1),
                {"n_estimators":[10,50,100],"max_samples":[0.6,0.8,1.0]}),
            "Random Forest": (RandomForestClassifier(random_state=1),
                {"n_estimators":[100,200,300],"max_depth":[3,5,10,None],
                 "min_samples_split":[2,5]}),
            "AdaBoost": (AdaBoostClassifier(random_state=1),
                {"n_estimators":[50,100,200],"learning_rate":[0.01,0.1,0.5,1.0]}),
            "GBM": (GradientBoostingClassifier(random_state=1),
                {"n_estimators":[100,200],"learning_rate":[0.05,0.1,0.2],
                 "max_depth":[3,5,7]}),
            "XGBoost": (XGBClassifier(random_state=1, eval_metric='logloss'),
                {"n_estimators":[100,200],"learning_rate":[0.05,0.1,0.2],
                 "max_depth":[3,5,7],"subsample":[0.7,0.9,1.0]}),
            "SVM": (SVC(kernel='rbf', random_state=1, probability=True),
                {"C":[0.1,1,10,100],"gamma":["scale","auto",0.01,0.001]}),
        }

        skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        top4_names = results_df["Model"].head(4).tolist()
        tuned_results = []
        for name in top4_names:
            if name not in param_grids:
                continue
            base_model, param_grid = param_grids[name]
            with st.spinner(f"Tuning {name}..."):
                search = RandomizedSearchCV(base_model, param_grid, n_iter=20,
                                            scoring='f1_weighted', cv=skf,
                                            random_state=1, n_jobs=-1)
                search.fit(x_train_final, y_train)
                bm   = search.best_estimator_
                ytp2 = bm.predict(x_train_final)
                yvp2 = bm.predict(x_val_final)
                tuned_results.append({
                    "Model":           name,
                    "Best Params":     str(search.best_params_),
                    "Train Accuracy":  round(accuracy_score(y_train,  ytp2), 4),
                    "Train F1":        round(f1_score(y_train,        ytp2, average=avg), 4),
                    "Train Recall":    round(recall_score(y_train,    ytp2, average=avg), 4),
                    "Train Precision": round(precision_score(y_train, ytp2, average=avg), 4),
                    "Val Accuracy":    round(accuracy_score(y_val,    yvp2), 4),
                    "Val F1":          round(f1_score(y_val,          yvp2, average=avg), 4),
                    "Val Recall":      round(recall_score(y_val,      yvp2, average=avg), 4),
                    "Val Precision":   round(precision_score(y_val,   yvp2, average=avg), 4),
                })

        tuned_df = pd.DataFrame(tuned_results).sort_values("Val F1", ascending=False)
        st.subheader("Hyperparameter Tuning — Top 4 Models")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("🟦 **Tuned Train Scores**")
            st.dataframe(tuned_df[["Model","Best Params","Train Accuracy","Train F1",
                                   "Train Recall","Train Precision"]].reset_index(drop=True),
                         use_container_width=True)
        with c2:
            st.markdown("🟧 **Tuned Validation Scores**")
            st.dataframe(tuned_df[["Model","Best Params","Val Accuracy","Val F1",
                                   "Val Recall","Val Precision"]].reset_index(drop=True),
                         use_container_width=True)

        best = tuned_df.iloc[0]
        st.success(f"🏆 Best Model: **{best['Model']}** | "
                   f"Val F1: {best['Val F1']} | Val Accuracy: {best['Val Accuracy']}")

    # ══════════════════════════════════════════════════════════════════════════
    # REGRESSION
    # ══════════════════════════════════════════════════════════════════════════
    if Problem_type == 'Regression':

        def model_perf_reg(model, predictors, target):
            pred = model.predict(predictors)
            return pd.DataFrame({
                "R2":   [round(r2_score(target,              pred), 4)],
                "MAE":  [round(mean_absolute_error(target,   pred), 4)],
                "MSE":  [round(mean_squared_error(target,    pred), 4)],
                "RMSE": [round(np.sqrt(mean_squared_error(target, pred)), 4)],
            })

        def model_performances_reg(models, x, y, xval, yval):
            st.subheader("Train vs Validation Performance")
            for i in range(0, len(models), 2):
                pair = models[i:i+2]
                cols = st.columns(len(pair))
                for col, (name, model) in zip(cols, pair):
                    model.fit(x, y)
                    with col:
                        st.markdown(f"**{name}**")
                        st.markdown("🟦 Train")
                        st.dataframe(model_perf_reg(model, x, y), use_container_width=True)
                        st.markdown("🟧 Validation")
                        st.dataframe(model_perf_reg(model, xval, yval), use_container_width=True)

            st.subheader("Cross Validation (Mean Scores)")
            kf      = KFold(n_splits=5, shuffle=True, random_state=1)
            scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            for i in range(0, len(models), 2):
                pair = models[i:i+2]
                cols = st.columns(len(pair))
                for col, (name, model) in zip(cols, pair):
                    scores = cross_validate(model, x, y, scoring=scoring,
                                            cv=kf, return_train_score=False)
                    with col:
                        st.markdown(f"**{name}** (CV)")
                        st.dataframe(pd.DataFrame({
                            'R2':   [round(scores['test_r2'].mean(), 4)],
                            'MAE':  [round(-scores['test_neg_mean_absolute_error'].mean(), 4)],
                            'MSE':  [round(-scores['test_neg_mean_squared_error'].mean(), 4)],
                            'RMSE': [round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), 4)],
                        }), use_container_width=True)

        def score_dif_reg(models, x, y, xval, yval):
            st.subheader("Train vs Validation Differences")
            for i in range(0, len(models), 2):
                pair = models[i:i+2]
                cols = st.columns(len(pair))
                for col, (name, model) in zip(cols, pair):
                    model.fit(x, y)
                    ytp = model.predict(x)
                    yvp = model.predict(xval)
                    d   = pd.DataFrame({
                        "Metric": ["R2","MAE","MSE","RMSE"],
                        "Train":  [round(r2_score(y,              ytp), 4),
                                   round(mean_absolute_error(y,   ytp), 4),
                                   round(mean_squared_error(y,    ytp), 4),
                                   round(np.sqrt(mean_squared_error(y, ytp)), 4)],
                        "Val":    [round(r2_score(yval,              yvp), 4),
                                   round(mean_absolute_error(yval,   yvp), 4),
                                   round(mean_squared_error(yval,    yvp), 4),
                                   round(np.sqrt(mean_squared_error(yval, yvp)), 4)],
                    })
                    d["Diff"] = (d["Train"] - d["Val"]).round(4)
                    with col:
                        st.markdown(f"**{name}**")
                        st.dataframe(d, use_container_width=True)

        par_models_reg = [
            ("Linear Regression", LinearRegression()),
            ("Ridge Regression",  Ridge()),
            ("Lasso Regression",  Lasso()),
        ]
        non_para_models_reg = [
            ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=3)),
            ("Decision Tree",       DecisionTreeRegressor(random_state=1)),
            ("Bagging",             BaggingRegressor(
                                        estimator=DecisionTreeRegressor(random_state=1),
                                        random_state=1)),
            ("Random Forest",       RandomForestRegressor(random_state=1)),
            ("AdaBoost",            AdaBoostRegressor(random_state=1)),
            ("GBM",                 GradientBoostingRegressor(random_state=1)),
            ("XGBoost",             XGBRegressor(random_state=1)),
            ("SVR",                 SVR(kernel='rbf', C=1.0, gamma='scale')),
        ]

        st.header("Parametric Models")
        model_performances_reg(par_models_reg, x_train_final, y_train, x_val_final, y_val)
        score_dif_reg(par_models_reg, x_train_final, y_train, x_val_final, y_val)

        st.header("Non-Parametric Models")
        model_performances_reg(non_para_models_reg, x_train_final, y_train, x_val_final, y_val)
        score_dif_reg(non_para_models_reg, x_train_final, y_train, x_val_final, y_val)

        st.header("Best Model Selection")
        from sklearn.model_selection import RandomizedSearchCV

        results = []
        for name, model in par_models_reg + non_para_models_reg:
            model.fit(x_train_final, y_train)
            ytp = model.predict(x_train_final)
            yvp = model.predict(x_val_final)
            results.append({
                "Model":      name,
                "Train R2":   round(r2_score(y_train,                   ytp), 4),
                "Train RMSE": round(np.sqrt(mean_squared_error(y_train, ytp)), 4),
                "Train MAE":  round(mean_absolute_error(y_train,        ytp), 4),
                "Val R2":     round(r2_score(y_val,                     yvp), 4),
                "Val RMSE":   round(np.sqrt(mean_squared_error(y_val,   yvp)), 4),
                "Val MAE":    round(mean_absolute_error(y_val,          yvp), 4),
            })
        results_df = pd.DataFrame(results).sort_values("Val R2", ascending=False)

        st.subheader("Model Comparison (Before Tuning)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("🟦 **Train Scores**")
            st.dataframe(results_df[["Model","Train R2","Train RMSE",
                                     "Train MAE"]].reset_index(drop=True),
                         use_container_width=True)
        with c2:
            st.markdown("🟧 **Validation Scores**")
            st.dataframe(results_df[["Model","Val R2","Val RMSE",
                                     "Val MAE"]].reset_index(drop=True),
                         use_container_width=True)

        param_grids_reg = {
            "Linear Regression": (LinearRegression(), {}),
            "Ridge Regression":  (Ridge(), {"alpha":[0.01,0.1,1,10,100,1000]}),
            "Lasso Regression":  (Lasso(), {"alpha":[0.001,0.01,0.1,1,10]}),
            "K-Nearest Neighbors": (KNeighborsRegressor(),
                {"n_neighbors":[3,5,7,9,11],"weights":["uniform","distance"],
                 "metric":["euclidean","manhattan"]}),
            "Decision Tree": (DecisionTreeRegressor(random_state=1),
                {"max_depth":[3,5,7,10,None],"min_samples_split":[2,5,10]}),
            "Bagging": (BaggingRegressor(estimator=DecisionTreeRegressor(random_state=1),
                                         random_state=1),
                {"n_estimators":[10,50,100],"max_samples":[0.6,0.8,1.0]}),
            "Random Forest": (RandomForestRegressor(random_state=1),
                {"n_estimators":[100,200,300],"max_depth":[3,5,10,None],
                 "min_samples_split":[2,5]}),
            "AdaBoost": (AdaBoostRegressor(random_state=1),
                {"n_estimators":[50,100,200],"learning_rate":[0.01,0.1,0.5,1.0]}),
            "GBM": (GradientBoostingRegressor(random_state=1),
                {"n_estimators":[100,200],"learning_rate":[0.05,0.1,0.2],
                 "max_depth":[3,5,7]}),
            "XGBoost": (XGBRegressor(random_state=1),
                {"n_estimators":[100,200],"learning_rate":[0.05,0.1,0.2],
                 "max_depth":[3,5,7],"subsample":[0.7,0.9,1.0]}),
            "SVR": (SVR(kernel='rbf'),
                {"C":[0.1,1,10,100],"gamma":["scale","auto",0.01,0.001],
                 "epsilon":[0.01,0.1,0.5]}),
        }

        kf         = KFold(n_splits=5, shuffle=True, random_state=1)
        top4_names = results_df["Model"].head(4).tolist()
        tuned_results = []
        for name in top4_names:
            if name not in param_grids_reg:
                continue
            base_model, param_grid = param_grids_reg[name]
            with st.spinner(f"Tuning {name}..."):
                if not param_grid:
                    base_model.fit(x_train_final, y_train)
                    bm, best_params = base_model, "N/A"
                else:
                    search = RandomizedSearchCV(base_model, param_grid, n_iter=20,
                                                scoring='r2', cv=kf,
                                                random_state=1, n_jobs=-1)
                    search.fit(x_train_final, y_train)
                    bm, best_params = search.best_estimator_, str(search.best_params_)
                ytp2 = bm.predict(x_train_final)
                yvp2 = bm.predict(x_val_final)
                tuned_results.append({
                    "Model":      name,
                    "Best Params":best_params,
                    "Train R2":   round(r2_score(y_train,                   ytp2), 4),
                    "Train RMSE": round(np.sqrt(mean_squared_error(y_train, ytp2)), 4),
                    "Train MAE":  round(mean_absolute_error(y_train,        ytp2), 4),
                    "Val R2":     round(r2_score(y_val,                     yvp2), 4),
                    "Val RMSE":   round(np.sqrt(mean_squared_error(y_val,   yvp2)), 4),
                    "Val MAE":    round(mean_absolute_error(y_val,          yvp2), 4),
                })

        tuned_df = pd.DataFrame(tuned_results).sort_values("Val R2", ascending=False)
        st.subheader("Hyperparameter Tuning — Top 4 Models")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("🟦 **Tuned Train Scores**")
            st.dataframe(tuned_df[["Model","Best Params","Train R2",
                                   "Train RMSE","Train MAE"]].reset_index(drop=True),
                         use_container_width=True)
        with c2:
            st.markdown("🟧 **Tuned Validation Scores**")
            st.dataframe(tuned_df[["Model","Best Params","Val R2",
                                   "Val RMSE","Val MAE"]].reset_index(drop=True),
                         use_container_width=True)

        best = tuned_df.iloc[0]
        st.success(f"🏆 Best Model: **{best['Model']}** | "
                   f"Val R2: {best['Val R2']} | Val RMSE: {best['Val RMSE']}")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL MODEL EVALUATION ON TEST DATA
    # ══════════════════════════════════════════════════════════════════════════
    st.header("Final Model Evaluation on Test Data")

    if Problem_type == 'Classification':
        best_name        = tuned_df.iloc[0]['Model']
        base_model, _    = param_grids[best_name]
        best_params_dict = eval(tuned_df.iloc[0]['Best Params'])

        with st.spinner(f"Retraining {best_name} on Train + Val..."):
            x_final_train = pd.concat([x_train_final, x_val_final], axis=0)
            y_final_train = np.concatenate([y_train, y_val])
            final_model   = base_model.set_params(**best_params_dict)
            final_model.fit(x_final_train, y_final_train)

            y_pred_test = final_model.predict(x_test_final)
            avg         = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'

            test_results = pd.DataFrame({
                "Metric": ["Accuracy","F1","Recall","Precision"],
                "Score":  [round(accuracy_score(y_test,  y_pred_test), 4),
                           round(f1_score(y_test,        y_pred_test, average=avg), 4),
                           round(recall_score(y_test,    y_pred_test, average=avg), 4),
                           round(precision_score(y_test, y_pred_test, average=avg), 4)]
            })
            st.subheader(f"Test Performance — {best_name}")
            st.dataframe(test_results)

            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            # ✅ Confusion Matrix & ROC side by side
            if len(np.unique(y_test)) == 2:
                from sklearn.metrics import roc_curve, auc
                prob_test   = final_model.predict_proba(x_test_final)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, prob_test)
                roc_auc     = auc(fpr, tpr)
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Confusion Matrix")
                    cm  = confusion_matrix(y_test, y_pred_test)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ConfusionMatrixDisplay(confusion_matrix=cm).plot(
                        ax=ax, colorbar=False, cmap='Blues')
                    ax.set_title(f"Confusion Matrix — {best_name}",
                                 fontweight='bold', color='red')
                    st.pyplot(fig)
                    plt.close(fig)
                with c2:
                    st.subheader("ROC Curve")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(fpr, tpr, color='darkorange', lw=2,
                            label=f"AUC = {roc_auc:.4f}")
                    ax.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("ROC Curve on Test Data", fontweight='bold', color='red')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.subheader("Confusion Matrix")
                cm  = confusion_matrix(y_test, y_pred_test)
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(confusion_matrix=cm).plot(
                    ax=ax, colorbar=False, cmap='Blues')
                ax.set_title(f"Confusion Matrix — {best_name}",
                             fontweight='bold', color='red')
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Validation vs Test Score Comparison")
            compare_df = pd.DataFrame({
                "Metric":     ["Accuracy","F1","Recall","Precision"],
                "Val Score":  [tuned_df.iloc[0]['Val Accuracy'],
                               tuned_df.iloc[0]['Val F1'],
                               tuned_df.iloc[0]['Val Recall'],
                               tuned_df.iloc[0]['Val Precision']],
                "Test Score": test_results["Score"].values,
            })
            compare_df["Difference"] = (compare_df["Val Score"] -
                                        compare_df["Test Score"]).round(4)
            st.dataframe(compare_df)

    if Problem_type == 'Regression':
        best_name     = tuned_df.iloc[0]['Model']
        base_model, _ = param_grids_reg[best_name]

        with st.spinner(f"Retraining {best_name} on Train + Val..."):
            x_final_train = pd.concat([x_train_final, x_val_final], axis=0)
            y_final_train = np.concatenate([y_train, y_val])

            if tuned_df.iloc[0]['Best Params'] == "N/A":
                final_model = base_model
            else:
                best_params_dict = eval(tuned_df.iloc[0]['Best Params'])
                final_model      = base_model.set_params(**best_params_dict)
            final_model.fit(x_final_train, y_final_train)

            y_pred_test  = final_model.predict(x_test_final)
            test_results = pd.DataFrame({
                "Metric": ["R2 Score","MAE","MSE","RMSE"],
                "Score":  [round(r2_score(y_test,                  y_pred_test), 4),
                           round(mean_absolute_error(y_test,        y_pred_test), 4),
                           round(mean_squared_error(y_test,         y_pred_test), 4),
                           round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4)]
            })
            st.subheader(f"Test Performance — {best_name}")
            st.dataframe(test_results)

            # ✅ Actual vs Predicted + Residual scatter side by side
            st.subheader("Plots")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Actual vs Predicted**")
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(y_test, y_pred_test, alpha=0.4, color='steelblue',
                           edgecolors='k', linewidths=0.2, s=20)
                ax.plot([y_test.min(), y_test.max()],
                        [y_test.min(), y_test.max()],
                        'r--', lw=2, label='Perfect Prediction')
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted", fontweight='bold', color='red')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            with c2:
                st.markdown("**Residuals vs Predicted**")
                residuals = y_test - y_pred_test
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(y_pred_test, residuals, alpha=0.4, color='coral',
                           edgecolors='k', linewidths=0.2, s=20)
                ax.axhline(0, color='red', linestyle='--', lw=2)
                ax.set_xlabel("Predicted Values")
                ax.set_ylabel("Residuals")
                ax.set_title("Residuals vs Predicted", fontweight='bold', color='red')
                st.pyplot(fig)
                plt.close(fig)

            # ✅ Residual distribution below
            st.subheader("Residual Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(residuals, kde=True, ax=ax, color='steelblue')
            ax.set_title("Residual Distribution", fontweight='bold', color='red')
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Validation vs Test Score Comparison")
            compare_df = pd.DataFrame({
                "Metric":     ["R2 Score","RMSE","MAE"],
                "Val Score":  [tuned_df.iloc[0]['Val R2'],
                               tuned_df.iloc[0]['Val RMSE'],
                               tuned_df.iloc[0]['Val MAE']],
                "Test Score": [round(r2_score(y_test,                  y_pred_test), 4),
                               round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
                               round(mean_absolute_error(y_test,        y_pred_test), 4)],
            })
            compare_df["Difference"] = (compare_df["Val Score"] -
                                        compare_df["Test Score"]).round(4)
            st.dataframe(compare_df)