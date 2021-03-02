###List Comp.

#1.ödev
#Numeric Değişken isimlerini büyük harfe çevir&başına NUM ekle.(numeric olmayanın da harfleri büyük olmalı)

import pandas as pd
import seaborn as sns
df=sns.load_dataset("car_crashes")
["NUM_"+col.upper() if df[col].dtype!="O" else col.upper() for col in df.columns]

#2.ödev
#İsminde "no" barındırmayan değişkenlerin sonuna "FLAG" yazınız.Tüm değişken isimleri büyük olmalı.

import pandas as pd
import seaborn as sns
df=sns.load_dataset("car_crashes")
[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]


#3.ödev
#Aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçerek yeni bir df oluşturunuz.

import seaborn as sns
df = sns.load_dataset("car_crashes")
og_list=["abbrev","no_previous"]
new_cols=[col for col in df.columns if col not in og_list]
new_cols
new_df=df[new_cols]
new_df.head()
