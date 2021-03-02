import pandas as pd
df=pd.read_csv("C:/Users/Lenovo/PycharmProjects/pythonProject4/titanic.csv")
df

def grab_col_names(df):
    cat_cols = [col for col in df.columns if df[col].dtype == "O"]
    num_cols=[col for col in df.columns if df[col].dtype != "O"]
    return cat_cols, num_cols

cat_cols, num_cols = grab_col_names(df)
grab_col_names(df)
print(cat_cols)
print(num_cols)

