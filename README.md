# LSTM Derin Öğrenme Modeli İle İleriki Zamanların Enerji Tüketimi Tahmini
![EfecanLogo](https://avatars.githubusercontent.com/u/66366306?s=100&u=dc5e6f5b4a05d07958d9a867b803760aa2b1613e&v=4)
### Projeyi star'larsan çok mutlu olurum.
## Ne yaptık?
- LSTM ve Derin Öğrenme Modeliyle Hollanda ülkesinin ileriki zamanlarda kullanabileceği enerji miktarını hesapladık.
- Proje ilk önce var olan verilerin kendisini tahmin edip grafiğe yansıtıyor. Bir sonraki kademede ileriki zamanlarda tahmini enerji kullanımının grafiğini yansıtıyor ve terminale logluyor. Hepsini değil. Özellikle demek istediğim; grafikte sadece birkaç ayı gösteriyor. Detaylandırabilirsiniz.
- Hazırladığım datada tüm yıllar içindeki aylar aynı. *Çünkü 1 yılı 12 aya bölerek aritmetik ortalamasını buldum*. Grafikte eğer çizgi düz giderse şaşırmayın.
- Proje basit bir proje kullanıcılar var olan modeli kendi ihtiyaçlarına göre uyarlayabilir/modele daha fazla katmanlar ekleyebilir.
- Derin modelleme için kullanılan paketler **TensorFlow ve Keras**'tır.

## Nasıl Kurarım?
- Terminalinize ```pip install -r requirements.txt``` yazarak uygun modülleri yükleyebilirsiniz.
- Python `3.7x` version gerekmektedir.

## Veri Kaynakçaları
- [Kaggle](kaggle.com/lucabasa/dutch-energy) ama verisetini projeye dahil etmedim. Sütunlar toplanıp ortalamalar baz alınarak csv dosyası oluşturuldu!
- Burda amaç zaten data değil derin öğrenme modeli örneği.
