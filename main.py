###########################################
#########                         #########
#####                                 #####
###             Hello World!            ###
###          Author: efecanxrd          ###
#####                                 #####
#########                         #########
###########################################

#Modüllerimizi ekleyelim. =Importing Modules=
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime, timedelta
import pandas as pd 
from depl_model import DeepLearningModel
import yaml
import os

#Verileri okuyalım ve zaman türünü belirtelim. =reading data and specifying the time type=
data = pd.read_csv('data.csv')
data['Datetime'] = [datetime.strptime(x, '%Y-%m') for x in data['Datetime']]

#Yinelenen verilerin ortalamasını alıyoruz! =We average the duplicate data! =
data = data.groupby('Datetime', as_index=False)['veri'].mean()

#Verileri sıralayalım. = (Let's sort the data) =
data.sort_values('Datetime', inplace=True)

#config dosyamızı tanımlayalım! =Importing Config File=
with open(f'{os.getcwd()}\\config.yml') as configfile:
    config = yaml.load(configfile, Loader=yaml.FullLoader)
    
#LSTM var olan dataların üstünden geçerek var olan datayı tahmin etsin. =Let LSTM estimate the existing data by going over the existing data.=
#Derin öğrenme dosyamızdaki modülü başlatalım. =Starting the deep learning node=
deepl = DeepLearningModel(
    data=data, #Datasetim(iz)
    Y_var='veri', #Enerji verileri =energy data=
    lag=config.get('lag'), #Epoch gecikmeleri. =Epoch delays=
    LSTM_layer_depth=config.get('LSTM_layer_depth'), #LSTM ağlarının derinliği =LSTM Layer Depth 64/128/256..=
    epochs=config.get('epochs'), #Epochs
    train_test_split=config.get('train_test_split') # Doğrulamak için kullanacağımız veriler =The data we will use to validate new data=
)

model = deepl.LSTModel()
yhat = deepl.predict()

if len(yhat) > 0:

  #Veri çerçevesini oluşturma = Data Frame =
    fc = data.tail(len(yhat)).copy()
    fc.reset_index(inplace=True)
    fc['forecast'] = yhat

    
#Tahminlerin grafiğini çizelim. =Graph the LSTM predictions.=
    plt.figure(figsize=(24, 16))
    for dtype in ['veri', 'forecast']:
        plt.plot(
            'Datetime',
            dtype,
            data=fc,
            label=dtype,
            alpha=0.8
        )
    print(fc)
    plt.legend()
    plt.grid()
    plt.show()   
    
#Gelecek verileri =Future Data=
#Modelin tüm verileri kullanarak oluşturulması ve ileride tahmin yapılması
deep_learner = DeepLearningModel(
    data=data, 
    Y_var='veri',
    lag=config.get('F.lag'),
    LSTM_layer_depth=config.get('F.LSTM_layer_depth'),
    epochs=config.get('F.epochs'),
    train_test_split=0 
)


deep_learner.LSTModel()

n_ahead = 168
yhat = deep_learner.predict_n_ahead(n_ahead)
yhat = [y[0][0] for y in yhat]

# Tahmin için veri çerçevesini oluşturma
fc = data.tail(400).copy() 
fc['type'] = 'original'

last_date = max(fc['Datetime'])
hat_frame = pd.DataFrame({
    'Datetime': [last_date + timedelta(days=x + 1) for x in range(n_ahead)], 
    'veri': yhat,
    'type': 'forecast'
})

fc = fc.append(hat_frame)
fc.reset_index(inplace=True, drop=True)

plt.figure(figsize=(12, 15))
for col_type in ['original', 'forecast']:
    plt.plot(
        'Datetime', 
        'veri', 
        data=fc[fc['type']==col_type],
        label=col_type
        )
print(fc[fc['type']==col_type])
plt.legend()
plt.grid()
plt.show()    
