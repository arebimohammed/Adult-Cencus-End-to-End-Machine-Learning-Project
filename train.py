import logging
logging.basicConfig()
import bentoml 
import joblib

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

cols = ['Age', 'Workclass', 'Fnlwgt', 'Education','EducationNum', 'MaritalStatus', 'Occupation', 'Relationship', 'Race',
       'Sex', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek', 'NativeCountry', 'Income'] 
df_train = pd.read_csv("Data/adult.data", delimiter=",", header=None, names=cols, index_col=False)
df_test = pd.read_csv("Data/adult.test", delimiter=",", header=None, names=cols, index_col=False, skiprows=1)
df_test['Income'] = df_test.Income.str.replace(".","")

df_final = pd.concat([df_train,df_test]).reset_index(drop=True)
categorical_cols = df_final.select_dtypes(include=['O']).columns
df_final[categorical_cols] = df_final[categorical_cols].apply(lambda x: x.str.strip())
df_final[df_final == '?'] = np.nan

df_final['Income'] = df_final.Income.replace({df_final.Income.unique()[0]: 0, df_final.Income.unique()[1]: 1})
df_final.drop(['EducationNum','Fnlwgt'], axis=1, inplace=True)

X = df_final.drop('Income', axis=1)
y = df_final['Income']


categorical_cols = df_final.select_dtypes(include=['O']).columns
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy='most_frequent', add_indicator = True)), 
                                          ("encoder", OneHotEncoder())])
preprocessor = ColumnTransformer(transformers=[("cat", categorical_transformer, categorical_cols)], remainder='passthrough')
preprocessor.fit(X)
X_model = preprocessor.transform(X)

model = XGBClassifier(n_estimators = 100, learning_rate = 0.09999999999999999, objective='binary:logistic')
model.fit(X_model, y)

logging.log(logging.INFO, "Saving...")

def save(model, bentoml_name, path):
    saved_model = bentoml.sklearn.save_model(bentoml_name, model,custom_objects={'preprocessor': preprocessor})    
    joblib.dump(model, path)
    return saved_model


saved_bento_model = save(model, "adult_xgboost", "models/adult_xgboost.joblib")
print(saved_bento_model)
logging.log(logging.INFO, "Done!")