# LSTM Derin Öğrenme Modeli İle İleriki Zamanların Enerji Tüketimi Tahmini
![EfecanLogo](https://avatars.githubusercontent.com/u/66366306?s=100&u=dc5e6f5b4a05d07958d9a867b803760aa2b1613e&v=4)
### Projeyi star'larsan çok mutlu olurum.
![XhW](https://i.imgur.com/qHAcfhX.gif)
## Ne yaptık?
- LSTM ve Derin Öğrenme Modeliyle Hollanda ülkesinin ileriki zamanlarda kullanabileceği enerji miktarını hesapladık.
- Proje ilk önce var olan verilerin kendisini tahmin edip grafiğe yansıtıyor. Bir sonraki kademede ileriki zamanlarda tahmini enerji kullanımının grafiğini yansıtıyor ve terminale logluyor. Hepsini değil. Özellikle demek istediğim; grafikte sadece birkaç ayı gösteriyor. Detaylandırabilirsiniz.
- Proje basit bir proje kullanıcılar var olan modeli kendi ihtiyaçlarına göre uyarlayabilir/modele daha fazla katmanlar ekleyebilir.
- Derin modelleme için kullanılan paketler **TensorFlow ve Keras**'tır.
- 2009 yılından itibaren başlıyor datamız ama veri setinin üzerinden giderken;
- Hazırladığım datada tüm yıllar içindeki aylar aynı. *Çünkü 1 yılı 12 aya bölerek aritmetik ortalamasını buldum*. Grafikte eğer çizgi düz giderse şaşırmayın.
- ![X](https://i.imgur.com/MmXgIHj.png)
- Gelecek verisi hazırlarken 2010 - 2020 arası yılları gösteriyor.
- ![x2](https://i.imgur.com/dcudIXr.png)
- Burada amaç derin öğrenme modelinin çalışma şeklini öğrenmek.
## Nasıl Kurarım?
- Terminalinize ```pip install -r requirements.txt``` yazarak uygun modülleri yükleyebilirsiniz.
- Python `3.7x` version gerekmektedir.

## Veri Kaynakçaları
- [Kaggle](https://kaggle.com/lucabasa/dutch-energy) ama verisetini projeye dahil etmedim. Sütunlar toplanıp ortalamalar baz alınarak csv dosyası oluşturuldu!
- Burda amaç zaten data değil derin öğrenme modeli örneği.
- ![XhW2](https://i.imgur.com/qHAcfhX.gif)
