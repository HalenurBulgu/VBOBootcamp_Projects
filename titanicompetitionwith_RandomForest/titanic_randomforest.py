##################################################
# RANDOM FOREST WITH KAGGLE TITANIC DATAS
##################################################

import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
from helpers.eda import *
from helpers.data_prep import *

dataframe_train = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/VBObootcamp/projects/datas/train.csv")
dataframe_test = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/VBObootcamp/projects/datas/test.csv")

dataframe_train.head()
dataframe_train.isnull().sum()
dataframe_test.isnull().sum()

# FEATURE ENGINEERING

#New Feaures

def titanic_prep(dataframe):
    #if cabin exist write 1, or 0
    dataframe["NEW_CABIN_BOOL"] = dataframe["Cabin"].isnull().astype('int')

    #new name feature
    dataframe["NEW_NAME_COUNT"] = dataframe["Name"].str.len()
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["Name"].apply(lambda x: len(str(x).split(" ")))
    dataframe["NEW_NAME_DR"] = dataframe["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    dataframe['NEW_TITLE'] = dataframe.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    dataframe["NEW_FAMILY_SIZE"] = dataframe["SibSp"] + dataframe["Parch"] + 1

    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

    dataframe.columns = [col.upper() for col in dataframe.columns]

    #DROP
    dataframe.drop(["TICKET", "NAME", "CABIN"], inplace=True, axis=1)


# MISSING VALUES

    #for new age features
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))

    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[(dataframe['AGE'] < 25), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 25) & (dataframe['AGE'] < 50), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 50), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 25), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 25) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 28), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 28) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    # Label encoding
    binary_columns = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and dataframe[col].dtypes == 'O']
    for col in binary_columns:
        dataframe = label_encoder(dataframe, col)

    # Rare encoding for:
    dataframe['NEW_TITLE'] = rare_encoder(dataframe[['NEW_TITLE']], 0.01)

    # One hot encoding:
    ohe_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'O' and 10 > len(dataframe[col].unique()) > 2]
    dataframe = one_hot_encoder(dataframe, ohe_columns, drop_first=False)

    return dataframe



####MACHINE LEARNING########

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

train_prep=titanic_prep(dataframe_train)
test_prep=titanic_prep(dataframe_test)

train_prep.isnull().sum()
test_prep.isnull().sum()
test_prep =test_prep.fillna(test_prep.mean())

X = train_prep.drop(["SURVIVED","PASSENGERID"], axis=1)
y = train_prep["SURVIVED"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)


#RANDOM FOREST
rf_model = RandomForestClassifier().fit(X_train,y_train)

# train hatası
y_pred = rf_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print("Accuracy:%.2f%%" % (accuracy * 100.0))
# Accuracy: %100


#test hatası
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:%.2f%%" % (accuracy * 100.0))
#%83.86

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 15],
             "n_estimators": [200, 300, 500],
             "min_samples_split": [2, 5, 8, 11]}


rf_model = RandomForestClassifier(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_


#######################################
# Final Model
#######################################

rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:%.2f%%" % (accuracy * 100.0))

#Accuracy:83.41%

print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test : ', X_test.shape)
print('y_test : ', y_test.shape)

"""X_train:  (668, 28)
y_train:  (668,)
X_test :  (223, 28)
y_test :  (223,)"""


#CONCAT TEST AND TRAIN

X_all= pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])

rf_model = RandomForestClassifier(random_state=42).fit(X_all, y_all)
y_pred = rf_model.predict(X_all)


rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 15],
             "n_estimators": [200, 300, 500],
             "min_samples_split": [2, 5, 8, 11]}

rf_model = RandomForestClassifier(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=1).fit(X_all, y_all)
rf_cv_model.best_params_

rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X_all, y_all)

y_pred = rf_tuned.predict(X_all)
accuracy = accuracy_score(y_all, y_pred)
print("Accuracy:%.2f%%" % (accuracy * 100.0))
#%91.02


kaggle_test=test_prep.drop(["PASSENGERID"], axis=1)
y_kaggle = rf_tuned.predict(kaggle_test)



kaggle_test["PassengerID"]=test_prep["PASSENGERID"]
kaggle_test["Survived"]=y_kaggle
kaggle_test.head()
kaggle_test.columns

kaggle_last=kaggle_test.drop(['PCLASS', 'SEX', 'AGE', 'SIBSP', 'PARCH', 'FARE', 'NEW_CABIN_BOOL',
       'NEW_NAME_COUNT', 'NEW_NAME_WORD_COUNT', 'NEW_NAME_DR',
       'NEW_FAMILY_SIZE', 'NEW_IS_ALONE', 'NEW_AGE_PCLASS', 'EMBARKED_C',
       'EMBARKED_Q', 'EMBARKED_S', 'NEW_TITLE_Master', 'NEW_TITLE_Miss',
       'NEW_TITLE_Mr', 'NEW_TITLE_Mrs', 'NEW_TITLE_Rare', 'NEW_AGE_CAT_mature',
       'NEW_AGE_CAT_senior', 'NEW_AGE_CAT_young', 'NEW_SEX_CAT_maturefemale',
       'NEW_SEX_CAT_maturemale', 'NEW_SEX_CAT_seniorfemale',
       'NEW_SEX_CAT_seniormale'], axis=1)

kaggle_last.head()



submission = pd.concat([test_prep['PASSENGERID'], pd.Series(y_kaggle)], axis=1)
submission.columns = ['PassengerId', 'Survived']

submission.to_csv('submission.csv', index=False)
submission.to_csv('C:/Users/Lenovo/OneDrive/Masaüstü/VBObootcamp/projects/datas/_halekagglesubmission.csv', index=False)



df_kaggle= pd.DataFrame({'PassengerId': test_prep['PASSENGERID'].loc[4], 'Survived': np.round(y_kaggle).astype(int) })
df_kaggle.reset_index(drop=True, inplace=True)
df_kaggle.to_csv('C:/Users/Lenovo/OneDrive/Masaüstü/VBObootcamp/projects/datas/halenur_kagglesubmission.csv', index=False)



submission.to_csv('submission.csv', index=False)
submission.to_csv('C:/Users/Lenovo/OneDrive/Masaüstü/VBObootcamp/projects/datas/_hkagglesubmission.csv', index=False)


###Visualizations

import matplotlib.pyplot as plt

sns.countplot(dataframe_train['Pclass'], hue=dataframe_train['Survived'])
plt.show()

sns.countplot(dataframe_train['Embarked'], hue=dataframe_train['Pclass'])
plt.show()

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_tuned, X_train)


