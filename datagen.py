import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from save_load import save
import matplotlib.pyplot as plt
import seaborn as sns


def extract_features(row, max_lag=3):
    percentiles = np.percentile(row, [10, 25, 50, 75, 90])

    feat = pd.Series({
    'mean' : np.mean(row),
    'median' : np.median(row),
    'mode' : row.mode().iloc[0] if not row.mode().empty else np.nan,
    'range' : np.ptp(row),
    'iqr' : np.subtract(*np.percentile(row, [75, 25])),
    'variance' : np.var(row),
    'std_dev': np.std(row),
    'cv' : np.std(row) / np.mean(row) if np.mean(row) != 0 else np.nan,
    'percentile_10' : percentiles[0],
    'percentile_25' : percentiles[1],
    'percentile_50' : percentiles[2],
    'percentile_75' : percentiles[3],
    'percentile_90' : percentiles[4],
    'autocorrelation' : row.autocorr()
    })

    # Calculate lag features
    for lag in range(1, max_lag + 1):
        if len(row) > lag:
            feat[f'lag_{lag}'] = row.shift(lag).iloc[lag] if len(row) > lag else 0
        else:
            feat[f'lag_{lag}'] = 0

    return feat


def chi_square_test(data, target):
    chi2_stats = []
    p_values = []

    for column in data.columns:
        contingency_table = pd.crosstab(data[column], target)
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi2_stats.append(chi2)
        p_values.append(p)

    # Bonferroni correction
    bonferroni_corrected_p = multipletests(p_values, method='bonferroni')[1]

    # Benjamini-Hochberg correction
    benjamini_hochberg_corrected_p = multipletests(p_values, method='fdr_bh')[1]

    results = pd.DataFrame({
        'Feature': data.columns,
        'Chi2': chi2_stats,
        'p-value': p_values,
        'Bonferroni p-value': bonferroni_corrected_p,
        'Benjamini-Hochberg p-value': benjamini_hochberg_corrected_p
    })

    return results


def anova_test(data, target):
    f_stats = []
    p_values = []

    for column in data.columns:
        groups = [data[column][target == t] for t in np.unique(target)]
        f_stat, p_value = stats.f_oneway(*groups)
        f_stats.append(f_stat)
        p_values.append(p_value)

    # Bonferroni correction
    bonferroni_corrected_p = multipletests(p_values, method='bonferroni')[1]

    # Benjamini-Hochberg correction
    benjamini_hochberg_corrected_p = multipletests(p_values, method='fdr_bh')[1]

    results = pd.DataFrame({
        'Feature': data.columns,
        'F-statistic': f_stats,
        'p-value': p_values,
        'Bonferroni p-value': bonferroni_corrected_p,
        'Benjamini-Hochberg p-value': benjamini_hochberg_corrected_p
    })

    return results


def combine_features(chi_square_results, anova_results, w_chisq=0.5, w_anova=0.5):
    # Standardize scores
    chi_square_results['Chi2_normalized'] = (chi_square_results['Chi2'] - chi_square_results['Chi2'].mean()) / \
                                            chi_square_results['Chi2'].std()
    anova_results['F-statistic_normalized'] = (anova_results['F-statistic'] - anova_results['F-statistic'].mean()) / \
                                              anova_results['F-statistic'].std()

    # Merge the results
    combined = pd.merge(chi_square_results, anova_results, on='Feature')

    # Calculate aggregate score
    combined['Aggregate_score'] = w_chisq * combined['Chi2_normalized'] + w_anova * combined['F-statistic_normalized']

    valid_features = combined.loc[combined['p-value_y'].notna(), 'Feature']

    return valid_features


def datagen():
    data = pd.read_csv('Dataset/Dataset.csv')

    label = data['Severity_None']

    categorical_cols = ['Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', \
                        'Nasal-Congestion', 'Runny-Nose', 'Severity_Mild', 'Severity_Moderate',
                        'Severity_None']


    plt.figure(figsize=(14, 10))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(3, 3, i)
        sns.countplot(data=data, x=col)
        plt.title(col)
        plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('Data Visualization/data.png')
    plt.show()

    data = data.drop(columns=['Severity_Mild', 'Severity_Moderate',  'Severity_None', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+'])

    correlation_matrix = data.corr()

    plt.figure(figsize=(16, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('Data Visualization/correlation_matrix.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x='Age_0-9', y='Tiredness')
    plt.title('Tiredness by Age Group')
    plt.savefig('Data Visualization/Violin Plot.png')
    plt.show()

    # Preprocessing
    # Handle missing values
    imputer = IterativeImputer()
    data_imputed = imputer.fit_transform(data)

    # Normalize the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_imputed)

    data_preprocessed = pd.DataFrame(data_normalized, columns=data.columns)

    # Feature Extraction
    features = data_preprocessed.apply(extract_features, axis=1)

    features_with_data = pd.concat([data_preprocessed, features], axis=1)

    features_with_data.to_csv('Dataset/features.csv', index=False)

    chi_square_results = chi_square_test(features_with_data, label)
    anova_results = anova_test(features_with_data, label)
    valid_features = combine_features(chi_square_results, anova_results)

    selected_features = features_with_data[valid_features]

    selected_features = np.array(selected_features)

    labels = np.array(label)

    train_sizes = [0.7, 0.8]
    for train_size in train_sizes:
        x_train, x_test, y_train, y_test = train_test_split(selected_features, labels, train_size=train_size)
        save('x_train_' + str(int(train_size * 100)), x_train)
        save('y_train_' + str(int(train_size * 100)), y_train)
        save('x_test_' + str(int(train_size * 100)), x_test)
        save('y_test_' + str(int(train_size * 100)), y_test)