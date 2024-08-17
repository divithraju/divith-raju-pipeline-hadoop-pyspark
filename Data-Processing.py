from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnProcessing").getOrCreate()

# Load customer data from HDFS
hdfs_path = "hdfs://localhost:50000/customer_churn"
customer_data = spark.read.csv(hdfs_path, header=True, inferSchema=True)

# Data Cleaning
cleaned_data = customer_data.dropna()  # Drop rows with missing values

# Handle Categorical Data: Convert categorical variables to numeric using StringIndexer
indexer = StringIndexer(inputCols=["gender", "contract_type"], outputCols=["gender_index", "contract_type_index"])
indexed_data = indexer.fit(cleaned_data).transform(cleaned_data)

# Feature Engineering
# Assemble features into a vector
feature_columns = ["age", "monthly_spend", "tenure", "gender_index", "contract_type_index"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data_with_features = assembler.transform(indexed_data)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaled_data = scaler.fit(data_with_features).transform(data_with_features)

# Split data into training and testing sets
(training_data, test_data) = scaled_data.randomSplit([0.7, 0.3])

# Train a Logistic Regression model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="churn")
model = lr.fit(training_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print("Model ROC-AUC:", roc_auc)

# Additional Metrics: Accuracy, Precision, Recall
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="churn", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)
print("Model Accuracy:", accuracy)

precision_evaluator = MulticlassClassificationEvaluator(labelCol="churn", metricName="precisionByLabel")
precision = precision_evaluator.evaluate(predictions)
print("Model Precision:", precision)

recall_evaluator = MulticlassClassificationEvaluator(labelCol="churn", metricName="recallByLabel")
recall = recall_evaluator.evaluate(predictions)
print("Model Recall:", recall)

# Save the trained model
model.save("hdfs://localhost:50000/customer_churn_model")

print("Model training complete and saved.")

# Stop the Spark session
spark.stop()
