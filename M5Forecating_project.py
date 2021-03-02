import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataframe=pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/kaggle_sets/kaggle2/sales_train_validation.csv")


def summary(dataframe,car_th=20,cat_th=10):
    print(f" Observations: {dataframe.shape[0]}")
    print(f" Variables: {dataframe.shape[1]}")
    print(f" Cat variables: {len([col for col in dataframe.columns if dataframe[col].dtypes == 'O'])},Cat but Car variables:{len([col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == 'O'])}")
    print(f" Num variables: {len([col for col in dataframe.columns if dataframe[col].dtypes != 'O'])},Num but Cat variables:{len([col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != 'O'])}")
summary(dataframe,car_th=20,cat_th=10)



def grab_cat_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    final_cat_cols = cat_cols + num_but_cat
    final_cat_cols = [col for col in final_cat_cols if col not in cat_but_car]
    return  cat_cols,num_but_cat,cat_but_car,final_cat_cols,
cat_cols, num_but_cat, cat_but_car, final_cat_cols = grab_cat_names(dataframe)


len(final_cat_cols)

dataframe[final_cat_cols]

dataframe[final_cat_cols].isnull().sum()

def cat_summary(dataframe,final_cat_cols):
    print(pd.DataFrame({"col_name": dataframe[final_cat_cols].value_counts(),"Ratio": 100 * dataframe[final_cat_cols].value_counts() / len(dataframe)}))
    print("######################################################")
    sns.countplot(x=dataframe[final_cat_cols], data=dataframe)
    plt.show()

for col in final_cat_cols:
    cat_summary(dataframe, col)

dataframe.store_id.count()
dataframe.store_id.CA_1.count()



num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O' and col not in num_but_cat]

dataframe[num_cols].head()


sns.scatterplot(x="d_1", y="d_400", data=dataframe)
plt.show()

dataframe[["d_1","d_400"]].corr()
dataframe.corr()

sns.lmplot(x="d_1", y="d_400", data=dataframe)
plt.show()

def target_summary_with_cat(dataframe, categorical_cols, target):
    for col in categorical_cols:
        if col not in target:
            print(pd.DataFrame({"TARGET_count": dataframe.groupby(col)[target].count()}), end="\n\n\n")
target_summary_with_cat(dataframe, final_cat_cols, "dept_id")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
target_summary_with_num(dataframe, "dept_id","d_1427")

