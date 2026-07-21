# %% [markdown]
# # Week 1 - Exploratory Data Analysis
# Customer Churn Prediction
#
# Run this file in VS Code: each `# %%` block is a runnable "cell"
# (click "Run Cell" above each block, or use Shift+Enter).
# Make sure your venv is selected as the interpreter first.


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# %%
# Load raw data directly here for exploration (not the cleaned version yet -
# we want to SEE the raw issues ourselves first)
df = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")
df.head()

# %%
df.info()

# %%
df.describe(include="all")

# %% [markdown]
# ## 1. Check target variable distribution (class imbalance)

# %%
churn_counts = df["Churn"].value_counts(normalize=True) * 100
print(churn_counts)

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Churn", palette=["#7fa1cb", "#e94560"])
plt.title("Churn Distribution")
plt.show()

# Note: You should see roughly 73% No / 27% Yes - this confirms
# class imbalance, which we'll handle later with SMOTE / class_weight.

# %% [markdown]
# ## 2. Check for data issues

# %%
# TotalCharges looks numeric but is stored as text - find why
df["TotalCharges"].apply(type).value_counts()

# %%
# Find the blank/bad values
pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum()

# %%
# Check missing values overall
df.isnull().sum()

# %% [markdown]
# ## 3. Churn vs Contract Type (classic churn insight)

# %%
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Contract", hue="Churn", palette=["#0f3460", "#e94560"])
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()

# Expect: month-to-month customers churn far more than 1-2 year contracts.
# Write this down as an insight for your README.

# %% [markdown]
# ## 4. Churn vs Tenure

# %%
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
             palette=["#0f3460", "#e94560"], bins=30)
plt.title("Tenure Distribution by Churn")
plt.show()

# Expect: churners cluster heavily in low-tenure (new) customers.

# %% [markdown]
# ## 5. Churn vs Monthly Charges

# %%
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn",
            fill=True, palette=["#0f3460", "#e94560"])
plt.title("Monthly Charges Distribution by Churn")
plt.show()

# %% [markdown]
# ## 6. Churn vs Tech Support / Online Security
# (Services that reduce churn are useful business insights)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(data=df, x="TechSupport", hue="Churn", ax=axes[0],
              palette=["#0f3460", "#e94560"])
sns.countplot(data=df, x="OnlineSecurity", hue="Churn", ax=axes[1],
              palette=["#0f3460", "#e94560"])
axes[0].set_title("Churn by Tech Support")
axes[1].set_title("Churn by Online Security")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Correlation heatmap (numeric features only)

# %%
numeric_df = df.copy()
numeric_df["TotalCharges"] = pd.to_numeric(numeric_df["TotalCharges"], errors="coerce")
numeric_df["Churn_binary"] = numeric_df["Churn"].map({"Yes": 1, "No": 0})

corr = numeric_df[["tenure", "MonthlyCharges", "TotalCharges", "Churn_binary"]].corr()

plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation with Churn")
plt.show()

# %% [markdown]
# ## Write your insights here (fill this in after running the cells above)
#
# 1. Month-to-month contract customers churn significantly more than 1-2 year contracts
# 2. Churn is heavily concentrated in customers with low tenure (< 10 months)
# 3. Customers without tech support or online security churn at a noticeably higher rate ...
# These become your README's "Key EDA Insights" section.
