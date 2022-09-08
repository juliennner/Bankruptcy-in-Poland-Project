import gzip
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wqet_grader
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from IPython.display import VimeoVideo
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


def wrangle(filename):

    # Open compressed file, load into dictionary
    with gzip.open(filename, "r") as f:
        data = json.load(f)

    # Load dictionary into DataFrame, set index
    df = pd.DataFrame().from_dict(data["data"]).set_index("company_id")

    return df


# In[3]:


df = wrangle("data/poland-bankruptcy-data-2009.json.gz")
print(df.shape)
df.head()

# Inspect DataFrame
df.info()

# Plot class balance
df["bankrupt"].value_counts(normalize=True).plot(
    kind="bar",
    xlabel="Bankrupt",
    ylabel="Frequency",
    title="Class Balance"
);


# Create boxplot
sns.boxplot(x="bankrupt", y="feat_27", data=df)
plt.xlabel("Bankrupt")
plt.ylabel("POA / financial expenses")
plt.title("Distribution of Profit/Expenses Ratio, by Class");

# Summary statistics for `feat_27`
df["feat_27"].describe().apply("{0:,.0f}".format)


# Plot histogram of `feat_27`
df["feat_27"].hist()
plt.xlabel("POA / financial expenses")
plt.ylabel("Count"),
plt.title("Distribution of Profit/Expenses Ratio");


# Create clipped boxplot
q1, q9 = df["feat_27"].quantile([0.1,0.9])
mask = df["feat_27"].between(q1, q9)
sns.boxplot(x="bankrupt", y="feat_27", data=df[mask])
plt.xlabel("Bankrupt")
plt.ylabel("POA / financial expenses")
plt.title("Distribution of Profit/Expenses Ratio, by Bankruptcy Status");




# Explore another feature
df["feat_60"].describe().apply("{0:,.0f}".format)


# In[4]:


sns.boxplot(x="bankrupt", y="feat_60", data=df)
plt.xlabel("Bankrupt")
plt.ylabel("Sales/Receivables")
plt.title("Distribution of Sales/Receivables, by Class");


# In[22]:


df["feat_60"].hist()
plt.xlabel("Sales/Receivables")
plt.ylabel("Frequency")
plt.title("Distribution of Sales/Receivables");


# In[23]:


q_1, q_9 = df["feat_60"].quantile([0.1,0.9])
mask_60 = df["feat_60"].between(q_1, q_9)
sns.boxplot(x="bankrupt", y="feat_60", data=df[mask_60])
plt.xlabel("Bankrupt")
plt.ylabel("Sales/Receivables")
plt.title("Distribution of Sales/Receivables, by Bankruptcy Status");



corr = df.drop(columns="bankrupt").corr()
sns.heatmap(corr);




# ## Split


target = "bankrupt"
X = df.drop(columns=target)
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ## Resample



under_sampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)
print(X_train_under.shape)
X_train_under.head()


# In[11]:


y_train_under.value_counts(normalize=True)


over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print(X_train_over.shape)
X_train_over.head()




y_train_over.value_counts(normalize=True)


# # Build Model

acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 4))


# Fit on `X_train`, `y_train`
model_reg = make_pipeline(
    SimpleImputer(strategy="median"),
    DecisionTreeClassifier(random_state=42)
)
model_reg.fit(X_train, y_train)

# Fit on `X_train_under`, `y_train_under`
model_under = make_pipeline(
    SimpleImputer(strategy="median"),
    DecisionTreeClassifier(random_state=42)
)
model_under.fit(X_train_under, y_train_under)

# Fit on `X_train_over`, `y_train_over`
model_over = make_pipeline(
    SimpleImputer(strategy="median"),
    DecisionTreeClassifier(random_state=42)
)
model_over.fit(X_train_over, y_train_over)


# ## Evaluate


for m in [model_reg, model_under, model_over]:
    acc_train = m.score(X_train, y_train)
    acc_test = m.score(X_test, y_test)

    print("Training Accuracy:", round(acc_train, 4))
    print("Test Accuracy:", round(acc_test, 4))

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(model_reg, X_test, y_test)


# In[20]:


ConfusionMatrixDisplay.from_estimator(model_over, X_test, y_test)


# Determine the depth of the decision tree in `model_over`.
depth = model_over.named_steps["decisiontreeclassifier"].get_depth()
print(depth)


# # Communicate

# Create a horizontal bar chart with the 15 most important features for `model_over`.

# Get importances
importances = model_over.named_steps["decisiontreeclassifier"].feature_importances_

# Put importances into a Series
feat_imp = pd.Series(importances, index=X_train_over.columns).sort_values()

# Plot series
feat_imp.tail(15).plot(kind="barh")
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("model_over Feature Importance");


# Save your model as `"model-5-2.pkl"`
with open("model-5-2.pkl", "wb") as f:
    pickle.dump(model_over, f)


# Load `"model-5-2.pkl"`
with open("model-5-2.pkl", "rb") as f:
    loaded_model = pickle.load(f)
print(loaded_model)
