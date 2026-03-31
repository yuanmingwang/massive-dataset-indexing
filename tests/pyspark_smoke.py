from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[1]").appName("smoke").getOrCreate()
sc = spark.sparkContext

print(sc.parallelize([1, 2, 3], 1).map(lambda x: x + 1).collect())

spark.stop()
