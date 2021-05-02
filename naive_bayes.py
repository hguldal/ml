
import os
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import StringIndexer

egitimDataYuzdesi=0.7

testDataYuzdesi=0.3

dataKonum=os.getcwd() + "/data/iris.csv"

dataOzellikAdlari=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]

dataSinifAdi="Species"

sc=SparkContext()

sqlContext = SQLContext(sc)

data = sqlContext.read.option("delimiter", ",").csv(dataKonum, header=True, inferSchema= True)

(egitimData, testData) = data.randomSplit([egitimDataYuzdesi, testDataYuzdesi], seed = 100)

sinifEtiketi = StringIndexer(inputCol=dataSinifAdi, outputCol="sinif")

ozellikListesi = VectorAssembler(inputCols=dataOzellikAdlari, outputCol="ozellikler")

nbSiniflandirici = NaiveBayes(smoothing=1.0, modelType="multinomial",labelCol="sinif", featuresCol="ozellikler")

pipeline = Pipeline(stages=[sinifEtiketi, ozellikListesi, nbSiniflandirici])

model = pipeline.fit(egitimData)

kestirimler = model.transform(testData)

kestirimVeSinif = kestirimler.select("prediction", "sinif").rdd

olcumler = MulticlassMetrics(kestirimVeSinif)

print(olcumler.confusionMatrix())

print(olcumler.accuracy)

