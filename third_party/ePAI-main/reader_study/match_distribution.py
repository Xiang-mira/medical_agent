'''
python -W ignore match_distribution.py --prediagnosis_csv prediagnosis.csv --normal_all_csv normal_all.csv --normal_csv normal.csv --target_size 100
'''

import os
import argparse
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def parse_spacing(spacing_str):
    # Convert string spacing like "(0.7421, 0.7421, 1.25)" to tuple of floats
    return tuple(map(float, spacing_str.strip('()').split(',')))

def frange(start, stop, step):
    """Generate a list of floats with fixed step size."""
    return [round(start + i * step, 6) for i in range(int((stop - start) / step) + 1)]

def match_metadata_distribution(positive_df, large_df, target_size, n_age_bins=10, n_spacing_bins=5, random_state=42):
    """
    Select exactly `target_size` samples from large_df that approximately match the metadata distribution of positive_df,
    considering sex, age, patient_status, contrast_enhanced, and spacing.
    """
    pos_df = positive_df.copy()
    large_df_copy = large_df.copy()

    # Parse spacing into three numeric columns
    for df in [pos_df, large_df_copy]:
        spacing_parsed = df['spacing'].apply(parse_spacing)
        df[['spacing_x', 'spacing_y', 'spacing_z']] = pd.DataFrame(spacing_parsed.tolist(), index=df.index)

    # Handle missing values for 'age' and spacing columns
    imputer = SimpleImputer(strategy='median')
    pos_df[['age']] = imputer.fit_transform(pos_df[['age']])
    large_df_copy[['age']] = imputer.transform(large_df_copy[['age']])

    spacing_cols = ['spacing_x', 'spacing_y', 'spacing_z']
    pos_df[spacing_cols] = imputer.fit_transform(pos_df[spacing_cols])
    large_df_copy[spacing_cols] = imputer.transform(large_df_copy[spacing_cols])

    # Discretize age and spacing into bins
    age_bin = KBinsDiscretizer(n_bins=n_age_bins, encode='ordinal', strategy='quantile')
    spacing_bin = KBinsDiscretizer(n_bins=n_spacing_bins, encode='ordinal', strategy='quantile')

    pos_df['age_bin'] = age_bin.fit_transform(pos_df[['age']]).astype(int)
    large_df_copy['age_bin'] = age_bin.transform(large_df_copy[['age']]).astype(int)

    spacing_cols = ['spacing_x', 'spacing_y', 'spacing_z']
    pos_df[[f'{col}_bin' for col in spacing_cols]] = spacing_bin.fit_transform(pos_df[spacing_cols]).astype(int)
    large_df_copy[[f'{col}_bin' for col in spacing_cols]] = spacing_bin.transform(large_df_copy[spacing_cols]).astype(int)

    # Define columns for stratification
    strat_cols = ['sex', 'age_bin', 'patient_status', 'contrast_enhanced',
                  'spacing_x_bin', 'spacing_y_bin', 'spacing_z_bin']
    pos_df['stratum'] = pos_df[strat_cols].astype(str).agg('-'.join, axis=1)
    large_df_copy['stratum'] = large_df_copy[strat_cols].astype(str).agg('-'.join, axis=1)

    # Target distribution
    stratum_distribution = pos_df['stratum'].value_counts(normalize=True)
    stratum_counts = (stratum_distribution * target_size).round().astype(int)

    sampled_df = []
    used_indices = set()

    for stratum, count in stratum_counts.items():
        group = large_df_copy[large_df_copy['stratum'] == stratum]
        available = group.loc[~group.index.isin(used_indices)]
        if len(available) >= count:
            sampled = available.sample(n=count, random_state=random_state)
        elif len(available) > 0:
            sampled = available
        else:
            continue
        used_indices.update(sampled.index)
        sampled_df.append(sampled)

    result_df = pd.concat(sampled_df)

    # Fill the gap if not enough samples were found
    if len(result_df) < target_size:
        remaining_needed = target_size - len(result_df)
        remaining_pool = large_df_copy.loc[~large_df_copy.index.isin(result_df.index)]

        remaining_pool['similarity_score'] = remaining_pool['stratum'].map(
            lambda s: stratum_distribution.get(s, 0)
        )

        filler = remaining_pool.sort_values(by='similarity_score', ascending=False).head(remaining_needed)
        result_df = pd.concat([result_df, filler])

    return result_df.drop(columns=[
        'age_bin', 'stratum', 'spacing_x', 'spacing_y', 'spacing_z',
        'spacing_x_bin', 'spacing_y_bin', 'spacing_z_bin', 'similarity_score'
    ], errors='ignore').reset_index(drop=True)

def plot_metadata_distributions(df1, df2, name1='prediagnosis', name2='normal', output_dir='distribution_plots'):
    os.makedirs(output_dir, exist_ok=True)

    categorical_cols = ['sex', 'patient_status', 'contrast_enhanced']
    continuous_cols = ['age', 'spacing_x', 'spacing_y', 'spacing_z']

    df1 = df1.copy()
    df2 = df2.copy()

    # Parse spacing
    df1[['spacing_x', 'spacing_y', 'spacing_z']] = pd.DataFrame(df1['spacing'].apply(parse_spacing).tolist())
    df2[['spacing_x', 'spacing_y', 'spacing_z']] = pd.DataFrame(df2['spacing'].apply(parse_spacing).tolist())

    # Plot continuous variables
    for col in continuous_cols:
        plt.figure(figsize=(6, 4))

        if col == 'age':
            min_age = int(min(df1[col].min(), df2[col].min()) // 10 * 10)
            max_age = int(max(df1[col].max(), df2[col].max()) // 10 * 10 + 10)
            bins = list(range(min_age, max_age + 1, 10))  # Bin width = 10
        elif col in ['spacing_x', 'spacing_y']:
            width = 0.1
            min_spacing = (min(df1[col].min(), df2[col].min()) // width) * width
            max_spacing = (max(df1[col].max(), df2[col].max()) // width + 1) * width
            bins = frange(min_spacing, max_spacing + width, width)
        elif col in ['spacing_z']:
            width = 0.25
            min_spacing = (min(df1[col].min(), df2[col].min()) // width) * width
            max_spacing = (max(df1[col].max(), df2[col].max()) // width + 1) * width
            bins = frange(min_spacing, max_spacing + width, width)

        sns.histplot(df1[col], color='blue', label=f'{name1} (hist)', bins=bins, stat='density', alpha=0.4, edgecolor='black')
        sns.histplot(df2[col], color='orange', label=f'{name2} (hist)', bins=bins, stat='density', alpha=0.4, edgecolor='black')

        # KDE curves
        sns.kdeplot(df1[col], color='blue', label=f'{name1} (kde)', linewidth=2)
        sns.kdeplot(df2[col], color='orange', label=f'{name2} (kde)', linewidth=2)

        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()

    # Plot categorical variables
    for col in categorical_cols:
        plt.figure(figsize=(6, 4))

        df1_counts = df1[col].value_counts(normalize=True).sort_index()
        df2_counts = df2[col].value_counts(normalize=True).sort_index()

        categories = sorted(set(df1_counts.index).union(df2_counts.index))
        df1_vals = [df1_counts.get(cat, 0) for cat in categories]
        df2_vals = [df2_counts.get(cat, 0) for cat in categories]

        x = range(len(categories))
        width = 0.35

        plt.bar([i - width/2 for i in x], df1_vals, width=width, label=name1, color='blue', edgecolor='black')
        plt.bar([i + width/2 for i in x], df2_vals, width=width, label=name2, color='orange', edgecolor='black')

        plt.xticks(ticks=x, labels=categories, rotation=45)
        plt.title(f'Distribution of {col}')
        plt.ylabel('Proportion')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Match metadata distribution between two datasets.')
    parser.add_argument('--prediagnosis_csv', type=str, default='prediagnosis.csv', required=True, help='Path to the CSV file with positive samples')
    parser.add_argument('--normal_all_csv', type=str, default='normal_all.csv', required=True, help='Path to the CSV file with large dataset')
    parser.add_argument('--normal_csv', type=str, default='normal.csv', help='Path to save the matched dataset')
    parser.add_argument('--target_size', type=int, default=8, help='Target size for the matched dataset')
    
    args = parser.parse_args()

    normal_all = pd.read_csv(args.normal_all_csv)
    prediagnosis = pd.read_csv(args.prediagnosis_csv)

    subset = match_metadata_distribution(prediagnosis, normal_all, target_size=args.target_size)

    subset.to_csv(args.normal_csv, index=False)

    # Plot and save metadata distributions
    plot_metadata_distributions(prediagnosis, subset)