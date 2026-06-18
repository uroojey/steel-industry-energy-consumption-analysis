# src/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set a nice theme for our plots
sns.set_theme(style="whitegrid")


def plot_correlations(df):
    """Show how features correlate with each other."""
    plt.figure(figsize=(10, 8))
    numeric_columns = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_columns.corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()


def plot_weekly_consumption(df):
    """Show average consumption grouped by day and hour."""
    heatmap_data = df.pivot_table(
        index='DayOfWeek', columns='Hour', values='Usage_kWh', aggfunc='mean')

    # Put days in chronological order
    days_order = ['Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False)
    plt.title('Average Energy Consumption by Day and Hour')
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred):
    """Plot actual usage vs predicted usage side-by-side."""
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.values[:100], label='Actual Usage')
    plt.plot(y_pred[:100], label='Predicted Usage', linestyle='--')
    plt.legend()
    plt.title('Actual vs Predicted Energy Consumption (First 100 Samples)')
    plt.show()


def plot_feature_importance(model, features):
    """Show which features were most important to the model."""
    importance_df = pd.DataFrame(
        {'Feature': features, 'Importance': model.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Key Drivers of Energy Consumption')
    plt.show()
