import pandas as pd
df=pd.read_csv("C:/Users/Lenovo/PycharmProjects/pythonProject4/titanic.csv")
df

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

def summary(df,car_th=20,cat_th=20):
    print(f" Observations: {df.shape[0]}")
    print(f" Variables: {df.shape[1]}")
    print(f" Cat variables: {len([col for col in df.columns if df[col].dtypes == 'O'])},Cat but Car variables:{len([col for col in df.columns if df[col].nunique() > car_th and df[col].dtypes == 'O'])}")
    print(f" Num variables: {len([col for col in df.columns if df[col].dtypes != 'O'])},Num but Cat variables:{len([col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes != 'O'])}")
summary(df,car_th=20,cat_th=20)

