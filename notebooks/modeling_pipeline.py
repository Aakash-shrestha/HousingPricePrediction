#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


#import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#import scikit learn stuffs
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# In[8]:


#load the dataset
housing_data = pd.read_csv("../data/AmesHousing.csv")
housing_data.head()


# In[9]:


#check for null columns
print(housing_data.isna().sum().sort_values(ascending=True))


# In[12]:


#drop the column having null values more than 80 percent
threshold = 0.8

housing_data = housing_data.loc[:, housing_data.isna().mean() < threshold]
list(housing_data.columns)


# In[13]:


#drop the columns which does not add much to the analysis
housing_data.drop(["Order", "PID"], axis=1, inplace=True, errors="ignore")


# In[14]:


#train test split data
X = housing_data.drop("SalePrice", axis=1)
y = housing_data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)


# In[22]:


#select numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
print(f"Numeric Features: {len(numeric_features)}")
print(f"Categorical Features: {len(categorical_features)}")


# In[24]:


#define steps for numeric and categorical features pipeline
numeric_steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
]

categorical_steps = [
    ('imputer', SimpleImputer(strategy="most_frequent")), #mode is used for cateogrical data
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
]


# In[26]:


#define the pipelines yayyy
numeric_transformer = Pipeline(numeric_steps)
categorical_transformer = Pipeline(categorical_steps)


# In[ ]:


#Column Transformer stuffs
#Column transformer allows diffrent transformer to apply for different kind of features
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# In[29]:


#Listing the models to compare
models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge' : Ridge()
}


# In[ ]:


#performing GridSearchCv and Cross Validation

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=21)

#define parameters for GridSearchCV
param_grid = {
    'Ridge': {'model__alpha': [0.1, 1, 10, 100]},
    'Lasso': {'model__alpha': [0.001, 0.01, 0.1, 1]}
}

#define a dictionary t jjo store the models
best_models = {}

for name, model in models.items():
    print(f"Training {name}")
    
    #define pipeline to be used for GridSearchCV estimator
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    grid = GridSearchCV(pipeline,
                        param_grid.get(name, {}),
                        cv=kf,
                        scoring='r2', #r squared
                        n_jobs=-1) #faster GridSearch
    
    grid.fit(X_train, y_train)
    best_models[name] = grid #stores the grid object for each model

print("Training Complete!")


# In[33]:


#evaluate the performance of each model

results = {}

for name, grid in best_models.items():
    y_pred = grid.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'Best Params': grid.best_params_,
        'Test MSE': mse, #lower better
        'Test R2': r2, #higher better
        'CV Score': grid.best_score_ #higher better
    }
    
    results_df = pd.DataFrame(results).T # T transposes the dataframe
    results_df.sort_values(by=['Test R2', 'CV Score', 'Test MSE'], ascending=[False, False, True])
    


# In[34]:


#print results df
print(results_df)


# *Since, Ridge has the lowest MSE, Highest R2 and highest CV score, it is the best !!*

# In[35]:


#store the best model 
import joblib
joblib.dump(best_models['Ridge'], '../models/ridge_best.pkl')


# In[ ]:


#plot the graph for acutal vs predicted values
y_pred = best_models['Ridge'].predict(X_test)
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ridge Regression: Actual vs Predicted Sale Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r') #draws diagonal line
plt.plot()

