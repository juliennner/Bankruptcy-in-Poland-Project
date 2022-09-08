import gzip
import json
import pickle

import ipywidgets as widgets
import pandas as pd
import wqet_grader
from imblearn.over_sampling import RandomOverSampler
from IPython.display import VimeoVideo
from ipywidgets import interact
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from teaching_tools.widgets import ConfusionMatrixWidget

wqet_grader.init("Project 5 Assessment")


# In[2]:


def wrangle(filename):
    with gzip.open(filename, "r") as f:
        data = json.load(f)
    df = pd.DataFrame().from_dict(data["data"]).set_index("company_id")
    return df


# In[3]:


df = wrangle("data/poland-bankruptcy-data-2009.json.gz")
print(df.shape)
df.head()


# ## Split

# In[4]:


target = "bankrupt"
X = df.drop(columns=target)
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ## Resample

# In[6]:


over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()


# # Build Model

# Calculate the baseline accuracy score for your model.

# In[7]:


acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 4))


# ## Iterate

# In[9]:


clf = make_pipeline(
    SimpleImputer(),
    GradientBoostingClassifier()
)
clf


# In[10]:


params = {
    "simpleimputer__strategy": ["mean", "median"],
    "gradientboostingclassifier__n_estimators": range(20, 31, 5),
    "gradientboostingclassifier__max_depth": range(2,5)
}
params

# In[12]:


model = GridSearchCV(
    clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)


# In[13]:


# Fit model to over-sampled training data
model.fit(X_train_over, y_train_over)


# In[14]:


results = pd.DataFrame(model.cv_results_)
results.sort_values("rank_test_score").head(10)


# In[15]:


# Extract best hyperparameters
model.best_params_


# ## Evaluate

# In[16]:


acc_train = model.score(X_train_over, y_train_over)
acc_test = model.score(X_test, y_test)

print("Training Accuracy:", round(acc_train, 4))
print("Validation Accuracy:", round(acc_test, 4))


# In[17]:


# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);

# In[18]:


# Print classification report
print(classification_report(y_test, model.predict(X_test)))


# In[19]:


model.predict(X_test)[:5]


# In[21]:


model.predict_proba(X_test)[:5,-1]


# In[22]:


c = ConfusionMatrixWidget(model, X_test, y_test)
c.show()


# In[23]:


c.show_eu()

# In[31]:


threshold = 0.2
y_pred_prob = model.predict_proba(X_test)[:,-1]
y_pred = y_pred_prob > threshold
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
print(f"Profit: €{tp * 100_000_000}")
print(f"Losses: €{fp * 250_000_000}")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False);


# In[32]:


def make_cnf_matrix(threshold):
    y_pred_prob = model.predict_proba(X_test)[:,-1]
    y_pred = y_pred_prob > threshold
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"Profit: €{tp * 100_000_000}")
    print(f"Losses: €{fp * 250_000_000}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False)

thresh_widget = widgets.FloatSlider(min=0, max=1, value=0.5, step=0.05)

interact(make_cnf_matrix, threshold=thresh_widget);


# In[33]:


# Save model
with open("model-5-4.pkl", "wb") as f:
    pickle.dump(model, f)


# In[34]:


get_ipython().run_cell_magic('bash', '', '\ncat my_predictor_lesson.py')



# In[35]:


# Import your module
from my_predictor_lesson import make_predictions

# Generate predictions
y_test_pred = make_predictions(
    data_filepath="data/poland-bankruptcy-data-2009-mvp-features.json.gz",
    model_filepath="model-5-4.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()
