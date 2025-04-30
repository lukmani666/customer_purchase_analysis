import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.distributions.empirical_distribution import ECDF
import warnings

# Setup
sns.set(style='whitegrid')
warnings.filterwarnings("ignore")
np.random.seed(42)

# Generate synthetic dataset
n = 1000
df = pd.DataFrame({
    'age': np.random.randint(18, 70, size=n),
    'gender': np.random.choice(['Male', 'Female'], size=n),
    'income': np.random.normal(50000, 15000, size=n).round(2),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], size=n),
    'loyalty_status': np.random.choice(['Bronze', 'Silver', 'Gold'], size=n, p=[0.5, 0.3, 0.2]),
    'purchase_frequency': np.random.poisson(5, size=n),
    'purchase_amount': np.random.exponential(scale=150, size=n).round(2),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], size=n),
    'promotion_usage': np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
    'satisfaction_score': np.random.randint(1, 11, size=n)
})

# Descriptive Statistics
desc_stats = df[['income', 'purchase_amount', 'satisfaction_score']].describe()

# Hypothesis Testing: Is there a difference in purchase amount by gender?
male_purchase = df[df['gender'] == 'Male']['purchase_amount']
female_purchase = df[df['gender'] == 'Female']['purchase_amount']
t_stat, p_value, _ = ttest_ind(male_purchase, female_purchase, usevar='unequal')

# Confidence Interval for satisfaction score of highest income quartile
q3_income = df['income'].quantile(0.75)
high_income = df[df['income'] >= q3_income]['satisfaction_score']
ci_mean = high_income.mean()
ci_std = high_income.std()
ci_n = len(high_income)
ci95 = stats.t.interval(0.95, ci_n-1, loc=ci_mean, scale=ci_std/np.sqrt(ci_n))

# Correlation Matrix
corr_matrix = df[['income', 'purchase_amount', 'satisfaction_score']].corr()

# Probability Distribution Fit for purchase_amount
ks_stat_norm, ks_pvalue_norm = stats.kstest(df['purchase_amount'], 'norm', args=(df['purchase_amount'].mean(), df['purchase_amount'].std()))
ks_stat_exp, ks_pvalue_exp = stats.kstest(df['purchase_amount'], 'expon', args=(df['purchase_amount'].min(), df['purchase_amount'].mean()))

# Outlier Detection
z_scores = np.abs(stats.zscore(df[['purchase_amount', 'satisfaction_score']]))
outliers = (z_scores > 3).any(axis=1)
outlier_df = df[outliers]

# Promotion usage vs satisfaction score
promotion_groups = df.groupby('promotion_usage')['satisfaction_score'].mean()

# Education vs purchase frequency across regions
edu_region_freq = df.groupby(['education', 'region'])['purchase_frequency'].mean().unstack()

# Top predictive features (based on correlation and visual inspection)
top_features = corr_matrix['purchase_amount'].abs().sort_values(ascending=False).index[1:4]

{
    "desc_stats": desc_stats,
    "t_test_result": {"t_stat": t_stat, "p_value": p_value},
    "confidence_interval_95": ci95,
    "correlation_matrix": corr_matrix,
    "distribution_fit": {
        "normal": {"ks_stat": ks_stat_norm, "p_value": ks_pvalue_norm},
        "exponential": {"ks_stat": ks_stat_exp, "p_value": ks_pvalue_exp}
    },
    "num_outliers": outlier_df.shape[0],
    "promotion_vs_satisfaction": promotion_groups,
    "education_region_freq": edu_region_freq,
    "top_predictive_features": top_features.tolist()
}



# Set up figure style
plt.style.use('seaborn-darkgrid')
figures = {}

# 1. Distribution of Purchase Amount
fig1, ax1 = plt.subplots()
sns.histplot(df['purchase_amount'], bins=30, kde=True, ax=ax1)
ax1.set_title('Distribution of Purchase Amount')
figures['purchase_amount_distribution'] = fig1

# 2. Boxplot of Purchase Amount by Gender
fig2, ax2 = plt.subplots()
sns.boxplot(x='gender', y='purchase_amount', data=df, ax=ax2)
ax2.set_title('Purchase Amount by Gender')
figures['purchase_amount_by_gender'] = fig2

# 3. Satisfaction Score by Promotion Usage
fig3, ax3 = plt.subplots()
sns.barplot(x='promotion_usage', y='satisfaction_score', data=df, ci='sd', ax=ax3)
ax3.set_title('Satisfaction Score by Promotion Usage')
ax3.set_xticklabels(['No Promo', 'Used Promo'])
figures['satisfaction_by_promotion'] = fig3

# 4. Correlation Heatmap
fig4, ax4 = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
ax4.set_title('Correlation Matrix')
figures['correlation_heatmap'] = fig4

# 5. Purchase Frequency by Education and Region
fig5, ax5 = plt.subplots(figsize=(10, 6))
edu_region_freq.plot(kind='bar', ax=ax5)
ax5.set_ylabel('Average Purchase Frequency')
ax5.set_title('Purchase Frequency by Education and Region')
figures['purchase_freq_edu_region'] = fig5

figures
