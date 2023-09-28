import polars as pl
import numpy as np
import pandas as pd
from polars import DataFrame, Series
from typing import Union
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import time


def polars_read_csv_lowered(data_file: Union[str, bytes]) -> DataFrame:
    """
    Import a CSV file using Polars and rename all columns to a lower case.

    Parameters:
    data_file (Union[str, bytes]): The path to the data file or the file-like
    object.

    Returns:
    DataFrame: Imported DataFrame with columns in lower case.
    """
    df = pl.read_csv(data_file)
    df = df.select([pl.col(name).alias(name.lower()) for name in df.columns])
    print(f'Imported {data_file} in shape: {df.shape}')
    return df


def change_x(df: DataFrame) -> DataFrame:
    """
    Update the 'status' column in a DataFrame by replacing 'X' with the last
    observed status for each 'sk_id_bureau'.

    Parameters:
    df (DataFrame): Input DataFrame with at least 'sk_id_bureau' and 'status'
    columns.

    Returns:
    DataFrame: Updated DataFrame with a new column 'updated_status'.
    """

    last_val = {}
    updated_status = []

    for row in df.iter_rows():
        sk_id = row[0]  # Assuming 'sk_id_bureau' is the first column
        status = row[2]  # Assuming 'status' is the third column

        if status == 'X':
            status = last_val.get(sk_id, '0')

        last_val[sk_id] = status
        updated_status.append(status)

    # Create a new Series with the updated status and add it to the DataFrame
    new_col = Series("updated_status", updated_status)

    return df.hstack([new_col])


def categorize_bur_credit_type(credit_type: str) -> str:
    """
    Categorizes the credit type into one of the most common categories in
    bureau dataset.
    """
    if credit_type is None:
        return 'other'
    credit_type = credit_type.lower()
    if 'consumer credit' in credit_type:
        return 'consumer credit'
    elif 'credit card' in credit_type:
        return 'credit card'
    elif 'car loan' in credit_type:
        return 'car loan'
    elif 'mortgage' in credit_type:
        return 'mortgage'
    elif 'microloan' in credit_type:
        return 'microloan'
    elif 'real estate loan' in credit_type:
        return 'real estate loan'

    return 'other'


def countplot_graph(title: str, df: pd.DataFrame, column: str) -> None:
    """
    Creates countplot of provided column in df in percent
    Inputs:
    df: df containing data
    column (str): column of countplot
    """
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=column, data=df)
    plt.title(title)

    # Annotate each bar with its corresponding percentage
    for p in ax.patches:
        pct = '{:.1f}%'.format(100 * p.get_height() / len(df))
        pct_x = p.get_x() + p.get_width() / 2
        pct_y = p.get_height() + 0.5
        ax.text(pct_x, pct_y, pct, ha='center', size=12)

    plt.show()


def proportion_graph(title: str, df: pd.DataFrame, column: str,
                     hue=None) -> None:
    """
    Creates a catplot with percentages for the provided column and hue in
    the DataFrame.

    Parameters:
    - title (str): The title of the plot.
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column to plot.
    - hue (str): The hue column.
    """

    if hue:
        temp_df = (df.groupby(column)[hue].value_counts(normalize=True).mul(
            100).rename('percent').reset_index())
    else:
        temp_df = (df[column].value_counts(normalize=True).mul(100).rename(
            'percent').reset_index().rename(columns={'index': column}))

    # Create the catplot
    plt.figure(figsize=(6, 4))
    g = sns.catplot(x=column, y='percent', hue=hue, kind='bar', data=temp_df)

    # Annotate each bar with its corresponding percentage
    for p in g.ax.patches:
        pct = str(p.get_height().round(1)) + '%'
        pct_x = p.get_x() + p.get_width() / 2
        pct_y = p.get_height() + 0.5
        g.ax.text(pct_x, pct_y, pct, ha='center', size=12)

    # Set title and axis labels
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Percent')

    plt.show()


def plot_distribution_by_target(title: str, df: pd.DataFrame, column: str,
                                hue: str, figsize: tuple = (12, 6),
                                print_pct: bool = False,
                                rotation: int = 0) -> None:
    """
    Creates a bar plot displaying the distribution of family statuses by the
    target variable.

    Parameters:
    - title (str): The title of the plot.
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column to plot.
    - hue (str): The hue column to group by, defaults to None.
    - figsize (tuple): The size of the figure, defaults to (12, 6).
    - print_pct (bool): Print percentages over each bar, default False.
    - rotation (int): Rotates xticks by degree. 0 is default

    Returns: graph
    """

    # Calculate the percentages
    temp_df = (
        df.groupby(hue)[column].value_counts(normalize=True).mul(100).rename(
            'percent').reset_index())

    # Create the bar plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=temp_df, x=column, y='percent', hue=hue)

    if print_pct:
        # Annotate each bar with its corresponding percentage
        for p in ax.patches:
            pct = str(p.get_height().round(1)) + '%'
            pct_x = p.get_x() + p.get_width() / 2
            pct_y = p.get_height() + 0.5
            ax.text(pct_x, pct_y, pct, ha='center', size=12)

    # Set title and axis labels
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Percent (%)')
    plt.xticks(rotation=rotation)
    plt.show()


def plot_kde_by_target(title: str, df: pd.DataFrame, column: str, hue: str,
                       log_scale: bool = False, xlim: tuple = None) -> None:
    """
    Creates a KDE plot by target for the specified DataFrame column.

    Parameters:
    - title (str): The title of the plot.
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column to plot.
    - hue (str): The target column.
    - log_scale (bool): Whether to use log scale.
    - xlim (tuple): x-axis limits.
    """
    plt.figure(figsize=(12, 6))

    # KDE plot with hue
    sns.kdeplot(data=df, x=column, hue=hue, fill=True, log_scale=log_scale,
                common_norm=False)

    # Set x-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    # Add title and labels
    plt.title(title)
    plt.xlabel(f"{column}{' (Log Scaled)' if log_scale else ''}")
    plt.ylabel('Density')

    plt.show()


def make_mi_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Compute Mutual Information scores."""
    X = X.copy()
    X = X.apply(lambda x: x.factorize()[0] if x.dtype == "object" else x)
    mi_scores = mutual_info_classif(X, y, random_state=0)
    mi_scores = (
        pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    )
    return mi_scores

def run_model_selection(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series,
                        preprocessor: Pipeline,
                        sample_size: float = 0.3,
                        sampler=None) -> pd.DataFrame:
    """
    Run a model selection process for a list of classifiers.

    Parameters:
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.
        preprocessor: A pipeline that performs the necessary preprocessing.
        sample_size: Fraction of the training data to use for model selection.
        sampler: The resampling method (like SMOTE, RandomUnderSampler).

    Returns:
        A DataFrame summarizing the performance of each classifier.
    """

    # Taking a smaller sample for model selection part and validation
    X_select_train, _, y_select_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=sample_size,
        stratify=y_train,
        random_state=42
    )

    classifiers = [
        LogisticRegression(n_jobs=-1),
        RandomForestClassifier(n_jobs=-1),
        DecisionTreeClassifier(),
        BaggingClassifier(n_jobs=-1),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        XGBClassifier(eval_metric='auc', n_jobs=-1, tree_method='gpu_hist'),
        LGBMClassifier(n_jobs=-1, metric='auc', device='gpu'),
        CatBoostClassifier(verbose=False, task_type="GPU")
    ]

    results = []

    for classifier in classifiers:
        start_time = time.time()

        # Define the pipeline
        steps = [("preprocessor", preprocessor)]

        if sampler is not None:
            steps.append(("sampler", sampler))

        steps.append(("classifier", classifier))

        pipeline = Pipeline(steps=steps)

        # Fit and predict
        model = pipeline.fit(X_select_train, y_select_train)
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]

        end_time = time.time()

        # Compute metrics
        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(y_val, predictions)
        recall = recall_score(y_val, predictions)
        f1 = f1_score(y_val, predictions)
        roc_auc = roc_auc_score(y_val, probabilities)
        time_elapsed = end_time - start_time

        results.append({
            'Classifier': type(classifier).__name__,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC AUC': roc_auc,
            'Time (s)': time_elapsed
        })

        print(f"{type(classifier).__name__} trained successfully in {time_elapsed:.3f} s")

    results_df = pd.DataFrame(results)
    return results_df


def conf_matrix_pred(y_test, y_pred):
    """
    Display a confusion matrix heatmap for the given ground truth labels and
    predicted labels.

    Parameters:
    - y_test : Ground truth (correct) target values.

    - y_pred : Estimated target values returned by a classifier.

    Returns:
    None. This function directly visualizes the confusion matrix using a heatmap.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cmap="Blues",
        square=True,
    )
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    all_sample_title = "Confusion matrix"
    plt.title(all_sample_title, size=15)
    plt.show()
    return cm

def calculate_profit_or_loss(conf_matrix,
                             mean_annuity,
                             interest_rate=0.1,
                             default_probability=0.04,
                             lgd=0.65):
    """
    Calculate and print profit or loss based on the confusion matrix and given assumptions.

    Parameters:
    - conf_matrix: Confusion matrix (2x2 NumPy array or list of lists)
    - mean_annuity: Mean annuity amount
    - interest_rate: Assumed average interest rate (default is 10% or 0.1)
    - default_probability: Assumed default probability (default is 4% or 0.04)
    - lgd: Assumed Loss Given Default (default is 65% or 0.65)

    Returns:
    - A dictionary containing the calculated values for loss per False Positive,
    loss per False Negative,
      total profit from True Negatives, total loss from False Positives, total
      loss from False Negatives,
      and total profit or loss.
    """
    conf_matrix = np.array(conf_matrix)
    true_negative, false_positive, false_negative, true_positive = conf_matrix.ravel()

    # Calculate loss per False Positive and False Negative
    loss_per_false_positive = mean_annuity * interest_rate
    loss_per_false_negative =  ((default_probability * lgd
                                 + (1 - default_probability) * interest_rate)
                                * mean_annuity)

    # Calculate total profit or loss
    profit_true_negative = loss_per_false_positive * true_negative
    total_loss_false_positive = loss_per_false_positive * false_positive
    total_loss_false_negative = loss_per_false_negative * false_negative
    total_profit_or_loss = profit_true_negative - total_loss_false_positive - total_loss_false_negative

    # Print the results
    print(f'Loss per False Positive prediction: ${loss_per_false_positive:.2f}')
    print(f'Loss per False Negative prediction: ${loss_per_false_negative:.2f}')
    print(f'Total Profit from True Negatives: ${profit_true_negative:.2f}')
    print(f'Total Loss from False Positives: ${total_loss_false_positive:.2f}')
    print(f'Total Loss from False Negatives: ${total_loss_false_negative:.2f}')
    print(f'Total Profit or Loss: ${total_profit_or_loss:.2f}')
