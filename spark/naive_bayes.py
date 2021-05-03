"""
Apache Spark ile Naive Bayes Sınıflandırıcı Kullanımı 
Hakan Güldal Trakya Üniversitesi
Edirne 2021 
"""

import os
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# **************************Veriseti İşlemleri*************************************************
dataKonum=os.getcwd() + "/data/iris.csv"

# Verisetindeki özelliklerin adları
dataOzellikAdlari=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]

# Verisetindeki sınıf etiketinin adı
dataSinifAdi="Species"

# Verisetinin ne kadarı eğitim için kullanılacak
egitimDataYuzdesi=0.7

# Verisetinin ne kadarı test için kullanılacak
testDataYuzdesi=0.3

# Spark Context ve SQL Context nesnelerini oluştur.
sc=SparkContext()

sqlContext = SQLContext(sc)

# CSV dosyasını oku, ayraç olarak , karakteri ve başlık satırları var
data = sqlContext.read.option("delimiter", ",").csv(dataKonum, header=True, inferSchema= True)

# Verisetini yüzdelik değerlerine göre 2 bölüme ayır 0: eğitim , 1:test
(egitimData, testData) = data.randomSplit([egitimDataYuzdesi, testDataYuzdesi], seed = 1234)

# Verisetinde sınıf etiketinin adı ve temsil adı
sinifEtiketi = StringIndexer(inputCol=dataSinifAdi, outputCol="sinif")

# Verisetindeki özelliklerin isimleri ve temsil adı
ozellikListesi = VectorAssembler(inputCols=dataOzellikAdlari, outputCol="ozellikler")

#Naive Bayes Sınıflandırıcını tanımla

nbSiniflandirici = NaiveBayes(smoothing=1.0, modelType="multinomial",labelCol="sinif", featuresCol="ozellikler")

# Pipeline oluştur
pipeline = Pipeline(stages=[sinifEtiketi, ozellikListesi, nbSiniflandirici])

# *******************Modeli Eğit*******************************************************************
model = pipeline.fit(egitimData)

# *******************Modeli Değerlendir ve Sonucu Göster*******************************************
kestirimler = model.transform(testData)

modelDegerlendirici = MulticlassClassificationEvaluator(labelCol="sinif", predictionCol="prediction", metricName="accuracy")

basariYuzdesi = modelDegerlendirici.evaluate(kestirimler)

print("Başarı Yüzdesi (Accuracy)=" + str(basariYuzdesi))

