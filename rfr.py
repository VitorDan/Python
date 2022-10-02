#RandomForestRegressor exemplo de codigo
#resuldados não tão bons devido a qualidade de dados
#alem de ser o primeiro teste com o algoritimo.


from cmath import sqrt
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
# from random import randint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error

from settings import DevConvfig as DevCfg

file_name = DevCfg['FILEPATH']
base =  pd.read_csv(file_name,sep=',')

X =  base[['Dia','Mes','Hora','Minuto','Segundo']].values
y = base['Multiplicador'].values
y = np.reshape(y, (-1,1))

# scaler =  MinMaxScaler(feature_range=(0,1))
scaler =  RobustScaler()


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
y_train,y_test = scaler.fit_transform(y_train), scaler.fit_transform(y_test)
model = RandomForestRegressor(random_state=122, n_jobs=100)
model.fit(X_train,y_train.ravel())
score = model.score(X_train,y_train)
scoreT = model.score(X_test,y_test)
y_predicr = model.predict(X_test)
mse = mean_squared_error(y_test,y_predicr.ravel())
rmse = sqrt(mse)

print('Train Score: ',score)
print('Test Score: ',scoreT)
print('MSE:',mse)
print('RMSE:',rmse)
pred = []
for i in range(0,60):
        pred.append([24,9,21,i,2])
pred = np.array(pred)
resul =  model.predict(pred)
resul = scaler.inverse_transform(np.reshape(resul, (-1,1)))
resul = np.reshape(resul, -1)

j=0
for i in range(len(pred)):
    if resul[i] > 2.9:
        j+=1
        print(f'{pred[i][-3]}:{pred[i][-2]}: {resul[i]:.2f}')
print(f'resultados: {j}')
