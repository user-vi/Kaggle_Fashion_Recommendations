from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizer

# StringIndexer https://www.kaggle.com/code/hanifansari93/starter-als-using-pyspark

# CountVectorizing https://stackoverflow.com/questions/50668577/expected-zero-arguments-for-construction-of-classdict-for-pyspark-ml-linalg-spa

# Use a NumPy array as a dense vector.
# dv1 = np.array([1.0, 0.0, 3.0])
# Create a SparseVector.
# sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])

spark = SparkSession.builder \
      .master("local") \
      .appName("test") \
      .enableHiveSupport() \
      .getOrCreate()

data = [
    ('user1', 'item1', 20),
    ('user1', 'item2', 30),
    ('user2', 'item2', 30),
    ('user2', 'item3', 30),
]
df = spark.createDataFrame(data, ["users", "items", "n"])
df.registerTempTable('data')


df2 = spark.sql("""
select users, collect_list(items) unigrams
from data
group by users
""")

# пропало часть данных, проверить гиперпараметры
# проверить как векторайзер сортирует униграммы
countVectors = CountVectorizer(inputCol="unigrams", outputCol="features", vocabSize=1, minDF=0.1)
model = countVectors.fit(df2)

dfA = model.transform(df2)
dfA.show(5, truncate=False)

dfB = model.transform(df2)
dfB.show(5, truncate=False)

brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0, numHashTables=3)
model = brp.fit(dfA)

model.transform(dfA).show(5, False)

model.approxSimilarityJoin(dfA, dfB, 1.5, distCol="EuclideanDistance")\
    .select(col("datasetA.users").alias("idA"),
            col("datasetB.users").alias("idB"),
            col("EuclideanDistance")).show()

