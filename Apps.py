import numpy as np
import pandas as pd

# untuk mendonload data saham
import pandas_datareader.data as web
import datetime
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt  # grafik

# metode LinearRegression
from sklearn.linear_model import LinearRegression
# Memprediksi nilai terdekat
from sklearn.neighbors import KNeigborsRegressor
# from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# forecast adalah jumlah hari


def predict(stock, start, end, forecast_out=7):
    try:
        # mengambil data saham lewat yahoo
        df = web.dataRender(stock, 'yahoo', start=start, end=end)
    except:
        print("some error, try another data or stock")
        exit

# adj close untuk melihat data real dari gabungan harga close dan volume
# menggunakan algoritma linear regresi untuk membandingkan fitur adj close dan volume saham
    dfreg = df.loc[:, ['adj close', 'volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['close'] * 100.0
    dfreg['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0

# drop mising value
    dfreg = dfreg.dropna()

# menggeser nilai adj close sebanyak 7 x
    dfreg['label'] = dfreg['Adj Close'].shift(-forecast_out)

    x = np.array(dfreg.drop(['label'], 1))
    x = preprocessing.scale(x)
# mengambil data x sebanyak 7
    x_lately = x[-forecast_out:]
    x = x[:-forecast_out]

    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    x_train = x
    y_train = y

    # untuk melihat nilai persen ter akurasi prediksi
    x_train, X_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, random_state=43)

    # fitur enginering di linear regresi
    clfreg = LinearRegression()
    clfreg.fit(x_train, y_train)

# poli quadratik regresion 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(x_train, y_train)

# poli quadratik regresion 3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(x_train, y_train)

    clfknn = KNeigborsRegressor(n_neighbors=1)
    clfknn.fit(x_train, y_train)

    # forecast_reg = clfreg.predict(x_lately)
    # forecast_poly2 = clfpoly2.predict(x_lately)
    # forecast_poly3 = clfpoly3.predict(x_lately)
    # forecast_knn = clfknn.predict(x_lately)

    forecast_reg = clfreg.predict(x_lately)
    forecast_poly2 = clfpoly2.predict(x_lately)
    forecast_poly3 = clfpoly3.predict(x_lately)
    forecast_knn = clfknn.predict(x_lately)

    dfreg['F_reg'] = np.nan
    dfreg['F_poly2'] = np.nan
    dfreg['F_poly3'] = np.nan
    dfreg['F_knn'] = np.nan

    last_data = dfreg.iloc[-1].name

    for i, k in enumerate(f_reg):
        next_date = last_data + datetime.timedelta(days=i+1)
        data = {'F_reg': k, 'F_poly2': f_poly2[i],
                'F_poly3': f_poly3[i], 'F_knn': f_knn[i]}
        dfreg = dfreg.append(pd.DataFrame(data, index=[next_date]))

    return (dfreg, f_reg, f_poly2, f_poly3, f_knn)

    # return (dfreg, forcast_reg, forcast_poly2, forcast_poly3, forcast_knn)
    # return (df, forcast_reg, forcast_poly2, forcast_poly3, forcast_knn)
if __name__ == '__main__':
    a = predict('BBCA_JK', datetime.datetime(
        2019, 1, 1), datetime.datetime.now())
    print(a)
