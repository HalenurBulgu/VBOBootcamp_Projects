import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Lenovo/OneDrive/Masaüstü/advertising.csv")
df.head()

# Model denklemlerini, tahmin fonksiyonlarını yazalım:

# b = 2.90, w1 = 0.04, w2 = 0.17, w3= 0.002
y1_hat = b + w1 * df['TV'] + w2 * df['radio'] + w3 * df['newspaper']

# b = 1.70, w1 = 0.09, w2 = 0.20, w3= 0.017
y2_hat = b + w1 * df['TV'] + w2 * df['radio'] + w3 * df['newspaper']

#y değerini tahmin edelim(y şapka değerini hesaplayalım):
df["y1_hat"] = 2.90 + 0.04 * df['TV'] + 0.17 * df['radio'] + 0.002 * df['newspaper']
df["y2_hat"] = 1.70 + 0.09 * df['TV'] + 0.20 * df['radio'] + 0.017 * df['newspaper']

df.head()

#mse değerini hesaplayalım:

y1_mse = np.mean((df['sales'] - df['y1_hat'])**2)
y2_mse = np.mean((df['sales'] - df['y2_hat'])**2)


#MAE değerini hesaplayalım:

y1_mae = np.mean(np.abs(df['sales'] - df['y1_hat']))
y2_mae = np.mean(np.abs(df['sales'] - df['y2_hat']))


#rmse değerini hesaplayalım:

y1_rmse = np.sqrt(np.mean((df['sales'] - df['y1_hat'])**2))
y2_rmse = np.sqrt(np.mean((df['sales'] - df['y2_hat'])**2))


print('RMS value for first set: %.2f'%y1_mse,",",'MAE value for first set: %.2f' %y1_mae,",",'RMSE value for first set: %.2f' %y1_rmse)
print('RMS value for second set: %.2f'%y2_mse,",",'MAE value for second set: %.2f' %y2_mae,",",'RMSE value for second set: %.2f' %y2_rmse)

#ilk modelin hatası ikinci modele göre daha düşüktür.
#İlk set daha iyi tahminler üretmektedir.

import matplotlib.pyplot as plt
x = df['sales']
y = df['y1_hat']
z= df['y2_hat']
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(x, x, label=f'sales')
plt.scatter(x, y, label=f'y1_hat')
plt.scatter(x, z, label=f'y2_hat')

# Plot
plt.title('Scatterplot')
plt.legend()
plt.show()
