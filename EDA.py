import pandas as pd
import streamlit as st
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


st.set_page_config(layout="wide")

st.title('Machine Learning Model')
img = Image.open(os.path.join(os.getcwd(), "static", "Machine learning.jpg"))
img = img.resize((1500, 600))  # (width, height) in pixels
st.image(img)

tab1, tab2, tab3, tab4 = st.tabs(['Tab1', 'Tab2', 'Tab3', 'Tab4'])
with tab1:
    Problem_type = st.selectbox("Problem Type", ['Classification', 'Regression'])
    if Problem_type is not None:
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "json"])
    else:
        st.write('Please select the problem_type')

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
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values
        })
        st.dataframe(info_df)

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = categorical_cols[df[categorical_cols].nunique() <= 10]
        st.session_state['categorical_cols'] = categorical_cols
        numerical_cols = df.select_dtypes(include=['number']).columns
        st.write(f"This dataset has {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")

        st.subheader('Checking whether the data has duplicate values')
        duplicates = df.duplicated().sum()
        st.write(duplicates)
        if duplicates == 0:
            st.write('Dataset doesnt have any duplicates')
        else:
            st.write(f'Dataset has {duplicates} number of duplicates in the dataframe')

        st.subheader("Checking whether data has null values")
        nulls = df.isnull().sum()
        st.write(nulls)
        if nulls.sum() == 0:
            st.write('Dataset doesnt have any null values')
        else:
            st.write(f'Dataset has {nulls.sum()} number of nulls in the dataframe')

        st.subheader('Value Counts')
        @st.fragment
        def value_counts_section(df):
            st.subheader('Value Counts')
            col = st.selectbox('Select a column which you want to see the value counts:', df.columns)
            st.write(f'value counts for columns: {col}')
            st.write(df[col].value_counts())

        # call it after all the other sections
        value_counts_section(df)

        st.session_state['df'] = df
        st.session_state['categorical_cols'] = categorical_cols
        st.session_state['numerical_cols'] = numerical_cols
        st.session_state['Target'] = Target

with tab2:
    # check session state before proceeding
    if 'df' not in st.session_state:
        st.warning("Please upload a dataset in Tab 1 first")
        st.stop()

    df = st.session_state['df']
    categorical_cols = st.session_state['categorical_cols']
    st.write("Categorical cols being used:", categorical_cols.tolist())  # ✅ debug line

    # retrieve from session state
    df = st.session_state['df']
    categorical_cols = st.session_state['categorical_cols']
    numerical_cols = st.session_state['numerical_cols']
    Target = st.session_state['Target']

    st.header("Univariate Analysis")
    st.subheader("Charts for Categorical columns")

    def bar_graph(data, feature, perc=True, n=None, figsize=(15, 5)):
        total = len(data[feature])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        sns.countplot(data=data, x=feature, palette='viridis', order=data[feature].value_counts().index[:n], ax=ax1)
        for p in ax1.patches:
            if perc == True:
                label = "{:.2f}%".format((p.get_height() / total * 100))
            else:
                label = p.get_height()
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax1.annotate(
                label, (x, y),
                size=8.5, ha='center', va='center',
                xytext=(0, 5), textcoords='offset points', fontweight='bold'
            )
        ax1.set_title(feature, fontweight='bold', fontsize=18, color='red')
        ax1.tick_params(axis='x', rotation=45)

        sns.boxplot(data=data, x=feature, palette='viridis', ax=ax2)
        ax2.set_title(f"Boxplot of {feature}", fontweight='bold', fontsize=18, color='red')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    for i in categorical_cols:
        fig = bar_graph(data=df, feature=i)
        st.pyplot(fig)
        plt.close(fig)

    def hist_box(data, feature, kde=True, figsize=(15, 5)):
        fig, (ax_hist, ax_box) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist)
        ax_hist.axvline(data[feature].mean(), color='red', linestyle='--', label='Mean')
        ax_hist.axvline(data[feature].median(), color='blue', linestyle='--', label='Median')
        ax_hist.legend()
        ax_hist.set_title('Distribution of ' + feature, color='red', fontweight='bold')

        sns.boxplot(data=data, x=feature, ax=ax_box)
        ax_box.set_title('Boxplot of ' + feature, color='red', fontweight='bold')

        plt.tight_layout()
        return fig

    st.subheader("Charts for Numerical columns")

    st.write("Numerical cols being used:", numerical_cols.tolist())
    for i in numerical_cols:
        fig = hist_box(data=df, feature=i)
        st.pyplot(fig)
        plt.close(fig)

    st.header("Bivariate Analysis")
    st.subheader("Charts for Categorical columns")

    def bivariate_count_plot(data, x, y, order=False):
        fig, ax = plt.subplots(figsize=(15, 5))
        if order == True:
            sns.countplot(data=data, x=x, hue=y, order=data[x].value_counts().index, palette='Set2', ax=ax)
        else:
            sns.countplot(data=data, x=x, hue=y, palette='Set2', ax=ax)
        for container in ax.containers:
            ax.bar_label(container)
        ax.set_title(x, fontweight='bold', fontsize=18, color='red')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        return fig

    for i in categorical_cols:
        fig = bivariate_count_plot(data=df, x=i, y=df[Target])
        st.pyplot(fig)
        plt.close(fig)
    
    st.subheader("Charts for Numerical columns")
    def distribution_plot_wrt_target(data, predictor, target):
        fig, axs = plt.subplots(2, 2, figsize=(15, 5))

        target_uniq = data[target].unique()

        axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
        sns.histplot(
            data=data[data[target] == target_uniq[0]],
            x=predictor,
            kde=True,
            ax=axs[0, 0],
            color="teal",
            stat="density",)

        axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
        sns.histplot(
            data=data[data[target] == target_uniq[1]],
            x=predictor,
            kde=True,
            ax=axs[0, 1],
            color="orange",
            stat="density",)
        axs[1, 0].set_title("Boxplot w.r.t target")
        sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")
        axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
        sns.boxplot(
            data=data,
            x=target,
            y=predictor,
            ax=axs[1, 1],
            showfliers=False,
            palette="Set2",)
        fig.suptitle(f"{predictor} vs {target}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
    for i in numerical_cols:
        fig = distribution_plot_wrt_target(data=df, predictor=i, target=Target)
        st.pyplot(fig)
        plt.close(fig)
    
    st.header("Multivariate Analysis")

    # Heatmap
    corr = df[numerical_cols].corr()  # ✅ correct way to get correlation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Heatmap for numerical variable", fontweight='bold', fontsize=20)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Boxplot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title('Boxplots', fontweight='bold', fontsize=30)
    ax.tick_params(axis='x', rotation=90)
    sns.boxplot(data=df[numerical_cols], ax=ax)  # ✅ only numerical cols
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Pairplot
    selected_cols = [col for col in df.columns if col in numerical_cols.tolist() + [Target]]  # ✅ dynamic instead of hardcoded
    g = sns.pairplot(df[selected_cols], hue=Target)
    g.fig.suptitle("Pairwise Scatterplots by Target", y=1.02, fontweight='bold')
    plt.tight_layout()
    st.pyplot(g.fig)
    plt.close()

    df = st.session_state['df']
    st.header("Tab 3")
    st.write("Add your content here")

with tab3:
    if 'df' not in st.session_state:
        st.warning("Please upload a dataset in Tab 1 first")
        st.stop()

    df = st.session_state['df']
    categorical_cols = st.session_state['categorical_cols']
    numerical_cols = st.session_state['numerical_cols']
    Target = st.session_state['Target']

    x = df.drop(Target, axis=1)
    y = df[Target]

    @st.fragment
    def split_section(x, y):
        test_size = st.slider("Select the % of test data", min_value=20, max_value=50)
        test_size_decimal = test_size / 100

        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=test_size_decimal, random_state=1)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, stratify=y_test, test_size=0.50, random_state=1)
        except ValueError:
            st.warning("Stratified split failed due to small class size — using regular split instead.")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_decimal, random_state=1)  # ✅ no stratify
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.50, random_state=1)

        st.write(f"Training size: {x_train.shape[0]} rows")
        st.write(f"Validation size: {x_val.shape[0]} rows")
        st.write(f"Test size: {x_test.shape[0]} rows")

        st.session_state['x_train'] = x_train
        st.session_state['x_test'] = x_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['x_val'] = x_val
        st.session_state['y_val'] = y_val

    split_section(x, y)

    if 'x_train' in st.session_state:
        x_train = st.session_state['x_train']
        x_val = st.session_state['x_val']
        x_test = st.session_state['x_test']

        # ✅ drop Target from cols if it accidentally ended up in x
        cat_cols = [col for col in categorical_cols if col in x_train.columns and col != Target]
        num_cols = [col for col in numerical_cols if col in x_train.columns and col != Target]

        # ✅ encode only categorical cols, keep numerical cols separate
        x_train_cat = pd.get_dummies(x_train[cat_cols], drop_first=True)
        x_val_cat = pd.get_dummies(x_val[cat_cols], drop_first=True)
        x_test_cat = pd.get_dummies(x_test[cat_cols], drop_first=True)

        # ✅ align val and test to train columns
        x_val_cat = x_val_cat.reindex(columns=x_train_cat.columns, fill_value=0)
        x_test_cat = x_test_cat.reindex(columns=x_train_cat.columns, fill_value=0)

        # ✅ scale numerical cols
        scaler = StandardScaler()
        x_train_num = pd.DataFrame(
            scaler.fit_transform(x_train[num_cols]),
            columns=num_cols, index=x_train.index
        )
        x_val_num = pd.DataFrame(
            scaler.transform(x_val[num_cols]),
            columns=num_cols, index=x_val.index
        )
        x_test_num = pd.DataFrame(
            scaler.transform(x_test[num_cols]),
            columns=num_cols, index=x_test.index
        )

        # ✅ combine encoded categorical and scaled numerical
        x_train_final = pd.concat([x_train_num, x_train_cat], axis=1)
        x_val_final = pd.concat([x_val_num, x_val_cat], axis=1)
        x_test_final = pd.concat([x_test_num, x_test_cat], axis=1)

        st.success("Data preprocessed successfully!")
        st.dataframe(x_train_final.head())

        # check imbalance
    y_train = st.session_state['y_train']
    y_val = st.session_state['y_val']
    y_test = st.session_state['y_test']

    st.subheader("Class Balance Check")
    balance = df[Target].value_counts(normalize=True) * 100
    st.write(balance)

    # threshold: if any class < 20% consider imbalanced
    is_imbalanced = balance.min() < 20

    if is_imbalanced:
        st.warning(f"Dataset is imbalanced! Minority class: {balance.min():.2f}%")
        use_smote = st.checkbox("Apply SMOTE to balance the data?")
        
        if use_smote:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=1)
            x_train_final, y_train = smote.fit_resample(x_train_final, y_train)
            st.success(f"SMOTE applied! New training size: {x_train_final.shape[0]} rows")
            st.write(pd.Series(y_train).value_counts(normalize=True) * 100)
    else:
        st.success(f"Dataset is balanced! Minority class: {balance.min():.2f}%")
    
    st.session_state['x_train_final'] = x_train_final
    st.session_state['x_val_final'] = x_val_final
    st.session_state['x_test_final'] = x_test_final

with tab4:
    if 'x_train' not in st.session_state:
        st.warning("Please complete preprocessing in Tab 3 first")
        st.stop()

    x_train_final = st.session_state['x_train_final']
    x_val_final = st.session_state['x_val_final']
    x_test_final = st.session_state['x_test_final']
    y_train = st.session_state['y_train']
    y_val = st.session_state['y_val']
    y_test = st.session_state['y_test']

    def model_performance_classification(model, predictors, target, threshold=0.5):
        prob_pred = model.predict_proba(predictors)[:, 1]  # ✅ use predict_proba for threshold
        class_pred = [1 if i >= threshold else 0 for i in prob_pred]

        df_perf = pd.DataFrame({
            "Accuracy": accuracy_score(target, class_pred),
            "Recall": recall_score(target, class_pred),
            "Precision": precision_score(target, class_pred),
            "F1": f1_score(target, class_pred),
        }, index=[0])
        return df_perf

    def model_performances(models, x, y, xval, yval):
        st.subheader("Train Performance")
        for name, model in models:
            model.fit(x, y)
            st.write(f"**{name}**")
            st.dataframe(model_performance_classification(model, x, y))

        st.subheader("Validation Performance")
        for name, model in models:
            st.write(f"**{name}**")
            st.dataframe(model_performance_classification(model, xval, yval))

        st.subheader("Cross Validation Performance (Mean Scores)")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for name, model in models:
            scores = cross_validate(
                model, x, y,
                scoring=['accuracy', 'precision', 'recall', 'f1'],
                cv=skf, return_train_score=False
            )
            mean_df = pd.DataFrame({
                'Accuracy': [scores['test_accuracy'].mean()],
                'Recall': [scores['test_recall'].mean()],
                'Precision': [scores['test_precision'].mean()],
                'F1 Score': [scores['test_f1'].mean()]
            })
            st.write(f"**{name}**")
            st.dataframe(mean_df)

    def AUC_ROC(models, x, y, xval, yval):
        st.subheader("AUC-ROC Scores")
        rows = []
        for name, model in models:
            model.fit(x, y)
            auc_train = roc_auc_score(y, model.predict_proba(x)[:, 1])       # ✅ predict_proba
            auc_val = roc_auc_score(yval, model.predict_proba(xval)[:, 1])   # ✅ predict_proba
            rows.append({"Model": name, "Train AUC": auc_train, "Val AUC": auc_val})
        st.dataframe(pd.DataFrame(rows))

    def score_dif(models, x, y, xval, yval):
        st.subheader("Training vs Validation Performance Differences")
        for name, model in models:
            model.fit(x, y)
            y_pred_train = model.predict(x)
            y_pred_val = model.predict(xval)

            diff_df = pd.DataFrame({
                "Metric": ["Accuracy", "Recall", "Precision", "F1"],
                "Train": [
                    accuracy_score(y, y_pred_train),
                    recall_score(y, y_pred_train),
                    precision_score(y, y_pred_train),
                    f1_score(y, y_pred_train)
                ],
                "Val": [
                    accuracy_score(yval, y_pred_val),
                    recall_score(yval, y_pred_val),
                    precision_score(yval, y_pred_val),
                    f1_score(yval, y_pred_val)
                ],
            })
            diff_df["Diff"] = diff_df["Train"] - diff_df["Val"]
            st.write(f"**{name}**")
            st.dataframe(diff_df)

    par_models = [
        ("Logistic Regression", LogisticRegression()),
        ("Naive Bayes", GaussianNB())
    ]

    model_performances(par_models, x_train_final, y_train, x_val_final, y_val)
    AUC_ROC(par_models, x_train_final, y_train, x_val_final, y_val)
    score_dif(par_models, x_train_final, y_train, x_val_final, y_val)