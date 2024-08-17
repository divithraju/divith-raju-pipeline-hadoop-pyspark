from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnIngestion").getOrCreate()

# Define HDFS path
hdfs_path = "hdfs://localhost:50000/customer_churn"

# Load CSV data from local directory
local_dataset_path = "/divithraju/hadoop/customer_churn_data/"
customer_data = spark.read.csv(local_dataset_path, header=True, inferSchema=True)

# Save the data to HDFS in CSV format
customer_data.write.mode("overwrite").csv(hdfs_path)

print("Customer data successfully ingested into HDFS at", hdfs_path)

# Stop the Spark session
spark.stop()
