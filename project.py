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

# MAGIC %md
# MAGIC 1) Import datasets

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

minuteMETs = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/minuteMETs.csv")

hourlySteps = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/hourlySteps.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC Questions:
# MAGIC - What are some trends in smart device usage?
# MAGIC - How could these trends apply to Bellabeat customers?
# MAGIC - How can these trends help influence Bellabeat marketing strategy?

# COMMAND ----------

#df = dailyHeartrate
#df = dailyActivity
#df = dailyCalories
#df = dailyIntensities
#df = dailySteps
#df = dailySleep
#df = weightLogInfo
df = minuteMETs

#df.printSchema()

display(df)

#cnt = df.count()
#display(cnt)

# COMMAND ----------

# MAGIC %md 
# MAGIC 2. Process and Data Cleaning

# COMMAND ----------

from pyspark.sql.functions import col, dayofweek, date_format, round

dailyActivityFormatted = dailyActivity.withColumn("TotalSteps", col("TotalSteps").cast("int"))

#dailyActivityFormatted.printSchema()

dailyActivityFormatted = dailyActivityFormatted\
    .withColumn("day_of_week", dayofweek("ActivityDate"))

dailyActivityFormatted = dailyActivityFormatted\
    .withColumn("day_of_week", date_format(col("ActivityDate"), "E"))

dailyActivityFormatted = dailyActivityFormatted\
    .withColumn("TotalDistance", round(col("TotalDistance"), 2))\
    .withColumn("TotalSteps", round(col("TotalSteps"), 2))\
    .withColumn("Calories", round(col("Calories"), 2))

minuteMETs=minuteMETs.withColumnRenamed('Activity','Date')

#display(dailyActivityFormatted)

# COMMAND ----------

#Modify data type of SleepDay

# COMMAND ----------

#Modify hourlyCalories_merged by splitting Time and date

# COMMAND ----------

#Modify hourlyIntensities_merged by splitting Time and date

# COMMAND ----------

#Number of Users
users = dailyActivity.select("Id").distinct()
display(users)

# COMMAND ----------

#Identify Missing Values 
missing_count = dailyActivity.filter(dailyActivity["Id"].isNull()).count()
print("Missing Count:", missing_count)

# COMMAND ----------

#Identifying for duplicates in DailyActivity

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Analyze

# COMMAND ----------

display(dailyActivityFormatted)

# COMMAND ----------

#Daily Average Analysis

from pyspark.sql.functions import avg, col

result = (
    dailyActivityFormatted
    .groupBy("day_of_week")
    .agg(avg("TotalSteps").alias("avg_steps"),
         avg("TotalDistance").alias("avg_distance"),
         avg("Calories").alias("avg_calories"))
)

result = result\
    .withColumn("avg_steps", round(col("avg_steps"), 2))\
    .withColumn("avg_distance", round(col("avg_distance"), 2))\
    .withColumn("avg_calories", round(col("avg_calories"), 2))

display(result)


# COMMAND ----------

# MAGIC %md
# MAGIC During which hour of the day were the more calories burned?

# COMMAND ----------

#make a chart

# COMMAND ----------

# MAGIC %md
# MAGIC Duration of each Activity and Calories Burned Per User

# COMMAND ----------

from pyspark.sql.functions import sum, col

# Perform the equivalent Spark operation
result = (
    dailyActivityFormatted
    .groupBy("Id")
    .agg(
        sum("TotalSteps").alias("total_steps"),
        sum("VeryActiveMinutes").alias("total_very_active_mins"),
        sum("FairlyActiveMinutes").alias("total_fairly_active_mins"),
        sum("LightlyActiveMinutes").alias("total_lightly_active_mins"),
        sum("Calories").alias("total_calories")
    )
)

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Average Steps Per Hours

# COMMAND ----------

display(hourlySteps)
#hourlySteps.printSchema()

# COMMAND ----------

from pyspark.sql.functions import split, col, to_timestamp, date_format


# Convert the string into a timestamp
hourlyStepsFormatted = hourlySteps.withColumn("ActivityHour", to_timestamp("ActivityHour", "M/d/yyyy h:mm:ss a"))

# Format the timestamp into the desired format
hourlyStepsFormatted = hourlyStepsFormatted.withColumn("ActivityHour", date_format("ActivityHour", "M/d/yyyy HH:mm"))

# Split the 'ActivityHour' column into two columns: 'Day' and 'Time'
hourlyStepsFormatted = hourlyStepsFormatted.withColumn("Day", split(col("ActivityHour"), " ")[0])
hourlyStepsFormatted = hourlyStepsFormatted.withColumn("Hour", split(col("ActivityHour"), " ")[1])

# Drop the original 'ActivityHour' column
hourlyStepsFormatted = hourlyStepsFormatted.drop("ActivityHour")

hourlyStepsFormatted = hourlyStepsFormatted.withColumnRenamed('StepTotal','Steps')

# Reorder the columns as per your requirement
hourlyStepsFormatted = hourlyStepsFormatted.select("Id", "Day", "Hour", "Steps")

display(hourlyStepsFormatted)

# COMMAND ----------

#Average Steps per Hour
from pyspark.sql.functions import col, avg

result = (
    hourlyStepsFormatted
    .groupBy("Hour")
    .agg(avg(col("Steps")).alias("avg_steps"))
    .orderBy("Hour")
)

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC - The average number of steps varies across different hours of the day, indicating distinct patterns in user activity.
# MAGIC - During the early morning hours (12 AM to 6 AM), the average steps are relatively low, suggesting that users tend to be less active during these hours, possibly due to sleeping.
# MAGIC - There is a noticeable increase in average steps from 6 AM to 10 AM, indicating a peak in activity during the morning hours. This could be attributed to activities such as morning walks or commutes.
# MAGIC - The average steps remain relatively consistent from 10 AM to 2 PM, suggesting that users maintain a certain level of activity during this midday period.
# MAGIC - The average steps decrease again in the late evening, suggesting reduced activity during the nighttime hours.
# MAGIC - Peak hours of user activity can be identified to target specific time slots for engagement, promotions, or notifications related to wellness activities.
# MAGIC - These insights provide a temporal understanding of user activity patterns throughout the day, allowing Bellabeat to tailor its product features or marketing strategies to align with users' daily routines.

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Metrics Comparison Over Time</h3><br>
# MAGIC This query can help understand if there are consistent patterns or if certain metrics are more influential in different periods.

# COMMAND ----------

from pyspark.sql.functions import avg

result3 = (
    dailyActivityFormatted
    .groupBy("ActivityDate")
    .agg(
        avg("TotalSteps").alias("avg_steps"),
        avg("TotalDistance").alias("avg_distance"),
        avg("Calories").alias("avg_calories")
    )
    .orderBy("ActivityDate")
)

display(result3)

import matplotlib.pyplot as plt 
import numpy as np 

x = result3.select('ActivityDate')
y1 = result3.select('avg_steps')
y2 = result3.select('avg_distance')
y3 = result3.select('avg_calories')
  
# plot lines 
graph1 = plt.plot(x, y1, label = "Steps") 
#graph2 = plt.plot(x, y2, label = "Distance") 
#graph3 = plt.plot(x, y3, label = "Calories") 

plt.legend() 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC minuteMETsNarrow_merged appears to contain information about METs (Metabolic Equivalents of Task) for different activity minutes.
# MAGIC
# MAGIC This table could be useful for exploring the intensity of physical activities undertaken by users and correlate it with other metrics like steps, distance, or calories burned.
# MAGIC
# MAGIC METs are a measure of the energy expenditure of physical activities.
# MAGIC
# MAGIC One MET is defined as the energy expenditure at rest, which is equivalent to sitting quietly.
# MAGIC
# MAGIC In the context of health and fitness tracking, METs are valuable because they provide a standardized way to measure and compare the intensity of different physical activities. Understanding METs allows to categorize activities based on their energy expenditure.
# MAGIC
# MAGIC Here's how METs are generally categorized:
# MAGIC
# MAGIC Low Intensity (1-3 METs): Activities such as sitting, standing, or casual walking.
# MAGIC Moderate Intensity (3-6 METs): Activities like brisk walking, cycling at a moderate pace, or light housework.
# MAGIC Vigorous Intensity (6+ METs): Activities that significantly raise your heart rate and breathing, such as running, cycling at a high speed, or intense exercise.

# COMMAND ----------

# MAGIC %md
# MAGIC Categorize users into different intensity levels

# COMMAND ----------

from pyspark.sql.functions import avg, col, when

# Perform the equivalent Spark operation
result = (
    minuteMETs
    .groupBy("Id")
    .agg(
        avg("METs").alias("avg_METs")
    )
    .withColumn(
        "intensity_category",
        when(col("avg_METs") <= 3, "Low Intensity")
        .when((col("avg_METs") > 3) & (col("avg_METs") <= 6), "Moderate Intensity")
        .when(col("avg_METs") > 6, "Vigorous Intensity")
        .otherwise("Unknown")
    )
    .select("Id", "avg_METs", "intensity_category")
)

display(result)

# COMMAND ----------

#display(minuteMETs)
#minuteMETs.printSchema

from pyspark.sql.functions import col, avg

# 1. Delete the column 'Minute'
minuteMETs = minuteMETs.drop("Minute")

# 2. Group by 'Date'
grouped_df = minuteMETs.groupBy("Id","Date")

# 3. Make a mean of 'METs'
result = grouped_df.agg(avg(col("METs")).alias("avg_METs"))

# 4. Order by user
result = result.orderBy("Id","Date")

result = result.filter("Id == '1624580081'")

display(result)
