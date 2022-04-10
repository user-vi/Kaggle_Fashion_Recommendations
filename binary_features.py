from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors

# StringIndexer https://www.kaggle.com/code/hanifansari93/starter-als-using-pyspark

# how to get vectors.sparse pyspark https://stackoverflow.com/questions/43809587/sparse-vector-pyspark

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

indexer = StringIndexer(inputCol="items", outputCol="categoryIndex")
df = indexer.fit(df).transform(df)
df.show()

pivoted = df.groupBy(["users"]).pivot("items").count().na.fill(0)
pivoted.show()

# VectorAssembler will choose more efficient representation depending on sparsity:
input_cols = [x for x in pivoted.columns if x != 'users']
result = (VectorAssembler(inputCols=input_cols, outputCol="features")
          .transform(pivoted)
          .select("users", "features"))
result.show()


brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0, numHashTables=3)
model = brp.fit(result)

model.transform(result).show(5, False)

model.approxSimilarityJoin(result, result, 1.5, distCol="EuclideanDistance")\
    .select(col("datasetA.users").alias("idA"),
            col("datasetB.users").alias("idB"),
            col("EuclideanDistance")).show()