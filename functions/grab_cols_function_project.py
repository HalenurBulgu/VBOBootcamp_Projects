import pandas as pd
dataframe=pd.read_csv("C:/Users/Lenovo/PycharmProjects/pythonProject4/titanic.csv")
dataframe
dataframe.shape[1]
target=input("hedef degiskeni giriniz")
def grab_cols(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and col not in target]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in target]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O" and col not in target]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O" and col not in target]
    final_cat_cols = cat_cols + num_but_cat
    final_cat_cols = [col for col in final_cat_cols if col not in cat_but_car and col not in target]
    target_col = [col for col in dataframe.columns if target in col]
    id_col=[col for col in dataframe.columns if "ID" in str(dataframe[col]).upper()]
    date_col = [col for col in dataframe.columns if dataframe[col].dtypes == "datetime64[ns]"]
    df_shape=len(num_cols+final_cat_cols+cat_but_car+target_col)-len(num_but_cat)
    return cat_cols,num_cols, num_but_cat, cat_but_car, final_cat_cols,target_col, id_col,date_col,df_shape
grab_cols(dataframe,cat_th=10, car_th=20)



