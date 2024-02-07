# Databricks notebook source
# MAGIC %md #FitBit Project

# COMMAND ----------

#the following code will delete all yours DBFS 
#display(dbutils.fs.ls("/FileStore/"))
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

# MAGIC %md #1) Import datasets

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
    .load("/FileStore/tables/dailyHeartrate.csv")

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

# MAGIC %md #2) Process and Data Cleaning

# COMMAND ----------

from pyspark.sql.functions import col, dayofweek, date_format, round

dailyActivityFormatted = dailyActivity

dailyActivityFormatted = dailyActivityFormatted\
    .withColumn("TotalSteps", col("TotalSteps").cast("int"))\
    .withColumn("WeekDay", dayofweek("ActivityDate"))\
    .withColumn("WeekDay", date_format(col("ActivityDate"), "E"))\
    .withColumn("TotalDistance", round(col("TotalDistance"), 2))\
    .withColumn("TotalSteps", round(col("TotalSteps"), 2))\
    .withColumn("Calories", round(col("Calories"), 2))\
    .withColumnRenamed('ActivityDate','Date')

#dailyActivityFormatted = dailyActivityFormatted.select('Id','Date','WeekDay','TotalSteps','TotalDistance','Calories')

#display(dailyActivityFormatted)
dailyActivityFormatted.printSchema()

# COMMAND ----------

#Check for missing-values and duplicates

from pyspark.sql.functions import count

missing_count = dailyActivityFormatted.filter(dailyActivityFormatted["Id"].isNull()).count()
print("Missing Count:", missing_count)

grouped_df = dailyActivityFormatted.groupBy("Id", "Date", "TotalSteps").agg(count("*").alias("Count"))
filtered_df = grouped_df.filter(grouped_df["Count"] > 1)
result_df = filtered_df.select("Id", "Date", "TotalSteps", "Count")
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##2.2) minuteMETs

# COMMAND ----------

# MAGIC %md
# MAGIC MET (Metabolic Equivalent of Task) is a physiological measure expressing the energy cost of physical activities. 
# MAGIC MET values are often used to estimate calorie expenditure during physical activities.

# COMMAND ----------

minuteMETsFormatted = minuteMETs

minuteMETsFormatted=minuteMETsFormatted\
  .withColumnRenamed('Activity','Date')\
  .withColumnRenamed('Minute','Time')

#display(minuteMETsFormatted)
minuteMETsFormatted.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ##2.3) dailySleep

# COMMAND ----------

from pyspark.sql.functions import to_date, split
from pyspark.sql.types import DateType
import pandas as pd

dailySleepFormatted = dailySleep

dailySleepFormatted = dailySleepFormatted\
    .withColumn("SleepDay", split(dailySleep.SleepDay, " ")[0])

dailySleepFormatted = dailySleepFormatted\
    .withColumn("SleepDay", to_date(dailySleepFormatted.SleepDay, "M/d/yyyy").cast(DateType()))

dailySleepFormatted = dailySleepFormatted\
    .withColumn("Weekday", dayofweek("SleepDay"))

dailySleepFormatted = dailySleepFormatted\
    .withColumn("Weekday", date_format(col("SleepDay"), "E"))

display(dailySleepFormatted)

# COMMAND ----------

# MAGIC %md
# MAGIC ##2.4) Other 

# COMMAND ----------

#Modify hourlyCalories by splitting Time and date

# COMMAND ----------

#Modify hourlyIntensities by splitting Time and date

# COMMAND ----------

#Number of Users
users = dailyActivity.select("Id").distinct()
display(users)

# COMMAND ----------

# MAGIC %md #3. Analyze

# COMMAND ----------

# MAGIC %md
# MAGIC Daily Average Analysis

# COMMAND ----------

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

# MAGIC %md
# MAGIC Average Steps per Hour

# COMMAND ----------

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

# MAGIC %md ##Metrics Comparison Over Time
# MAGIC This query can help understand if there are consistent patterns or if certain metrics are more influential in different periods.

# COMMAND ----------

from pyspark.sql.functions import avg

result3 = (
    dailyActivityFormatted
    .groupBy("Date")
    .agg(
        avg("TotalSteps").alias("avg_steps"),
        avg("TotalDistance").alias("avg_distance"),
        avg("Calories").alias("avg_calories")
    )
    .orderBy("Date")
)

display(result3)

'''import matplotlib.pyplot as plt 
import numpy as np 

x = result3.select('Date')
y1 = result3.select('avg_steps')
y2 = result3.select('avg_distance')
y3 = result3.select('avg_calories')
  
graph1 = plt.plot(x, y1, label = "Steps") 
graph2 = plt.plot(x, y2, label = "Distance") 
graph3 = plt.plot(x, y3, label = "Calories") 

plt.legend() 
plt.show()'''

# COMMAND ----------

# MAGIC %md
# MAGIC - minuteMETs appears to contain information about METs (Metabolic Equivalents of Task) for different activity minutes.
# MAGIC
# MAGIC   This table could be useful for exploring the intensity of physical activities undertaken by users and correlate it with other metrics like steps, distance, or calories burned.
# MAGIC
# MAGIC   - METs are a measure of the energy expenditure of physical activities.
# MAGIC
# MAGIC   - One MET is defined as the energy expenditure at rest, which is equivalent to sitting quietly.
# MAGIC
# MAGIC   - In the context of health and fitness tracking, METs are valuable because they provide a standardized way to measure and compare the intensity of different physical activities. Understanding METs allows to categorize activities based on their energy expenditure.
# MAGIC
# MAGIC - Here's how METs are generally categorized:
# MAGIC   - Low Intensity (1-3 METs): Activities such as sitting, standing, or casual walking.
# MAGIC   - Moderate Intensity (3-6 METs): Activities like brisk walking, cycling at a moderate pace, or light housework.
# MAGIC   - Vigorous Intensity (6+ METs): Activities that significantly raise your heart rate and breathing, such as running, cycling at a high speed, or intense exercise.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Categorize users into different intensity levels

# COMMAND ----------

from pyspark.sql.functions import avg, col, when

result = (
    minuteMETs
    .groupBy("Id")
    .agg(
        avg("METs").alias("avg_METs")
    )
    .withColumn(
        "intensity_category",
        when(col("avg_METs") <= 3, "LOW")
        .when((col("avg_METs") > 3) & (col("avg_METs") <= 6), "MEDIUM")
        .when(col("avg_METs") > 6, "HIGH")
        .otherwise("Unknown")
    )
    .select("Id", "avg_METs", "intensity_category")
)

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC The output shows the average METs and the corresponding intensity category for each user.
# MAGIC
# MAGIC In this case, all users are categorized as "Vigorous Intensity" based on the provided threshold values.
# MAGIC
# MAGIC This indicates that the average METs for each user fall within the range associated with vigorous intensity activities.
# MAGIC
# MAGIC For users categorized under "Vigorous Intensity," Bellabeat could consider providing tailored recommendations and features to support and enhance their vigorous intensity activities.
# MAGIC
# MAGIC Users can be offered:
# MAGIC - Specialized Workouts: Offer workout programs or sessions specifically designed for vigorous intensity exercises. This could include high-intensity interval training (HIIT) routines, advanced cardio workouts, and strength training programs.
# MAGIC - Performance Tracking: Enhance the app's tracking capabilities for vigorous activities. Provide detailed insights into users' performance during high-intensity exercises.
# MAGIC - Motivational Content: Create motivational content and challenges targeted at users engaging in vigorous activities.
# MAGIC - Community Engagement: Foster a sense of community among users with similar activity levels. This could include forums, groups, or challenges specifically for those engaging in vigorous intensity workouts, allowing users to share experiences and tips.
# MAGIC - Health and Safety Tips: Offer health and safety tips related to vigorous exercise. Provide information on proper warm-ups, cool-downs, hydration, and recovery strategies to ensure users stay safe and maximize the benefits of their workouts.
# MAGIC - Integration with Wearables: If users are using Bellabeat's smart wellness products during vigorous activities, ensure seamless integration with wearables to capture accurate and real-time data. This can enhance the overall user experience.
# MAGIC - Personalized Recommendations: Leverage the collected data to provide personalized recommendations for users engaging in vigorous intensity activities. This could include suggested workout routines, recovery strategies, and nutritional guidance tailored to individual preferences and goals.

# COMMAND ----------

# MAGIC %md 
# MAGIC ##Intensity of Activities
# MAGIC Exploring the distribution of METs to understand the range of activity intensities recorded by the devices.
# MAGIC Identifying peak MET values and correlating them with specific activities or time periods.

# COMMAND ----------

from pyspark.sql.functions import col

result = (
    minuteMETs
    .groupBy("METs")
    .count()
    .orderBy("METs")
)

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Sleep analysys
# MAGIC Analyzing the average total minutes asleep to understand the typical sleep duration.<br>
# MAGIC Looking for trends or patterns in sleep data over time.

# COMMAND ----------

from pyspark.sql.functions import avg

dailySleepFormatted2 = dailySleepFormatted

# Drop unnecessary columns and calculate average TotalTimeInBed grouped by Weekday
sleepmean = dailySleepFormatted2\
    .drop("Id", "SleepDay", "TotalSleepRecords", "TotalTimeInBed")\
    .groupBy("Weekday")\
    .agg(round(avg("TotalMinutesAsleep")).alias("AvgSleepMinutes"))

display(sleepmean)

# COMMAND ----------

# MAGIC %md ##User classification by training status
# MAGIC Segmenting users based on their activity and sleep patterns. This can help identify different user groups with distinct behaviors.

# COMMAND ----------

from pyspark.sql.functions import when, col, avg



joined_df = dailyActivity.alias("A")\
    .join(dailySleep.alias("S"), col("A.ActivityDate") == col("S.SleepDay"), "inner")

display(joined_df)

# Group by Id and calculate average TotalSteps and TotalMinutesAsleep
user_segments = joined_df\
    .groupBy("A.Id").agg(avg("TotalSteps").alias("AvgSteps"), avg("TotalMinutesAsleep").alias("AvgMinutesAsleep"))



#display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Average METs for single user

# COMMAND ----------

#display(minuteMETs)
#minuteMETs.printSchema

from pyspark.sql.functions import col, avg

minuteMETs = minuteMETs.drop("Minute")
grouped_df = minuteMETs.groupBy("Id","Date")
result = grouped_df.agg(avg(col("METs")).alias("avg_METs"))
result = result.orderBy("Id","Date")
result = result.filter("Id == '1624580081'")

display(result)
