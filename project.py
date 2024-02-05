# Databricks notebook source
# MAGIC %md #FitBit Project

# COMMAND ----------

#the following code will delete all yours DBFS 
#display(dbutils.fs.ls("/FileStore"))
#dbutils.fs.rm("/FileStore/tables", recurse=True)

# COMMAND ----------

#dbutils.fs.cp("/heartrate.csv", "FileStore/tables/a.csv")

# Specify the relative path to the file in the same directory
#relative_path = "/heartrate.csv"

# Read the CSV file into a DataFrame
#df = spark.read.csv(relative_path, header=True, inferSchema=True)


#relative_path = "subdirectory/your_file.csv"
#df = spark.read.csv(relative_path, header=True, inferSchema=True)

# COMMAND ----------

inferSchema = "true"

dailyActivity = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/dailyActivity.csv")

dailyCalories = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/dailyCalories.csv")

dailyHeartrate = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/heartrate.csv")

dailyIntensities = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/dailyIntensities.csv")

dailySteps = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/dailySteps.csv")

dailySleep = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/dailySleep.csv")

weightLogInfo = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/weightLogInfo.csv")

# COMMAND ----------

display(dailyHeartrate)
