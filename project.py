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

# MAGIC %md
# MAGIC ##2.1) dailyActivity

# COMMAND ----------

from pyspark.sql.functions import col, dayofweek, date_format, round

dailyActivityFormatted = dailyActivity

dailyActivityFormatted = dailyActivityFormatted\
    .withColumn("TotalSteps", col("TotalSteps").cast("int"))\
    .withColumn("WeekDay", dayofweek("ActivityDate"))\
    .withColumn("WeekDay", date_format(col("ActivityDate"), "E"))\
    .withColumnRenamed('ActivityDate','Date')
    #.withColumn("TotalDistance", round(col("TotalDistance"), 2))\
    #.withColumn("TotalSteps", round(col("TotalSteps"), 2))\
    #.withColumn("Calories", round(col("Calories"), 2))\

#dailyActivityFormatted = dailyActivityFormatted.select('Id','Date','WeekDay','TotalSteps','TotalDistance','Calories')

#display(dailyActivityFormatted)

# COMMAND ----------

#Check for missing-values and duplicates

from pyspark.sql.functions import count

missing_count = dailyActivityFormatted.filter(dailyActivityFormatted["Id"].isNull()).count()
print("Missing Count:", missing_count)

grouped_df = dailyActivityFormatted.groupBy("Id", "Date", "TotalSteps").agg(count("*").alias("Count"))
filtered_df = grouped_df.filter(grouped_df["Count"] > 1)
result_df = filtered_df.select("Id", "Date", "TotalSteps", "Count")

#display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##2.2) minuteMETs
# MAGIC MET (Metabolic Equivalent of Task) is a physiological measure expressing the energy cost of physical activities. 
# MAGIC MET values are often used to estimate calorie expenditure during physical activities.

# COMMAND ----------

minuteMETsFormatted = minuteMETs

minuteMETsFormatted=minuteMETsFormatted\
  .withColumnRenamed('Activity','Date')\
  .withColumnRenamed('Minute','Time')

#display(minuteMETsFormatted)
#minuteMETsFormatted.printSchema()

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

dailySleepFormatted = dailySleepFormatted.drop('TotalSleepRecords')

#display(dailySleepFormatted)

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
#display(users)

# COMMAND ----------

# MAGIC %md #3. Analyze

# COMMAND ----------

# MAGIC %md
# MAGIC Daily Average Analysis

# COMMAND ----------

from pyspark.sql.functions import avg, col

q1 = (
    dailyActivityFormatted
    .groupBy("WeekDay")
    .agg(avg("TotalSteps").alias("avg_steps"),
         avg("TotalDistance").alias("avg_distance"),
         avg("Calories").alias("avg_calories"))
)

q1 = q1\
    .withColumn("avg_steps", round(col("avg_steps"), 2))\
    .withColumn("avg_distance", round(col("avg_distance"), 2))\
    .withColumn("avg_calories", round(col("avg_calories"), 2))

display(q1)

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

q2 = (
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

display(q2)

# COMMAND ----------

# MAGIC %md
# MAGIC Average Steps Per Hours

# COMMAND ----------

#TODO: move to 'Process and Data Cleaning section'

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

#display(hourlyStepsFormatted)

# COMMAND ----------

from pyspark.sql.functions import col, avg

q3 = (
    hourlyStepsFormatted
    .groupBy("Hour")
    .agg(avg(col("Steps")).alias("avg_steps"))
    .orderBy("Hour")
)

display(q3)

#TODO: make a chart

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
import matplotlib.pyplot as plt
import numpy as np

q4 = (
    dailyActivityFormatted
    .groupBy("Date")
    .agg(
        round(avg("TotalSteps")).alias("avg_steps"),
        round(avg("TotalDistance")*1000).alias("avg_distance"),
        round(avg("Calories")).alias("avg_calories")
    )
    .orderBy("Date")
    .toPandas()
)

x = q4.Date
y1 = q4.avg_steps
y2 = q4.avg_distance
y3 = q4.avg_calories

#set size
plt.figure(figsize=(15, 3))
plt.tight_layout()
  
graph1 = plt.plot(x, y1, label = "Steps")
graph2 = plt.plot(x, y2, label = "Distance")
graph3 = plt.plot(x, y3, label = "Calories")
plt.xticks(rotation=90)
ax = plt.gca()

#set x tick density
ax.set_xticks( [* range(int(ax.get_xticks()[0])-1, int(ax.get_xticks()[-1]), int( (ax.get_xticks()[-1] - ax.get_xticks()[0])/(len(ax.get_xticks())-1) / 3 )) ] )

#set y tick density
ax.set_yticks( [* range(int(ax.get_yticks()[0]), int(ax.get_yticks()[-1])+1, int( (ax.get_yticks()[-1] - ax.get_yticks()[0])/(len(ax.get_yticks())-1) / 2 )) ] )

plt.legend() 
plt.show()

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

#TODO: to fix it ==> all intensities are HIGH

from pyspark.sql.functions import avg, col, when

q5 = (
    minuteMETsFormatted
    .groupBy("Id")
    .agg(avg("METs").alias("avg_METs"))
    .withColumn("intensity_category",
        when(col("avg_METs") <= 3, "LOW")
        .when((col("avg_METs") > 3) & (col("avg_METs") <= 6), "MEDIUM")
        .when(col("avg_METs") > 6, "HIGH")
        .otherwise("Unknown")
    )
    .select("Id", "avg_METs", "intensity_category")
)

display(q5)

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

q6 = (
    minuteMETsFormatted
    .groupBy("METs")
    .count()
    .orderBy("METs")
)

display(q6)

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

#TODO: use matplot to plot the chart

# COMMAND ----------

# MAGIC %md
# MAGIC Exploring day-to-day variability in both physical activity and sleep metrics. Identifying trends or unusual events that might impact users' routines.

# COMMAND ----------

#Sleep Trend Analysis

from pyspark.sql import SparkSession

joined_df = dailyActivityFormatted\
    .join(dailySleepFormatted, dailyActivityFormatted["Date"] == dailySleepFormatted["SleepDay"], "inner")

q7 = joined_df\
    .select("SleepDay", "TotalMinutesAsleep")\
    .orderBy("SleepDay")

display(q7)

# COMMAND ----------

# MAGIC %md ##User classification by training status
# MAGIC Segmenting users based on their activity and sleep patterns. This can help identify different user groups with distinct behaviors.

# COMMAND ----------

from pyspark.sql.functions import when, col, avg

activityAndSleep = dailyActivityFormatted.alias('A')\
    .join(dailySleepFormatted.alias('S'), col("A.Date") == col("S.SleepDay"), "inner")

# Group by Id and calculate average TotalSteps and TotalMinutesAsleep
userSegments = activityAndSleep\
    .groupBy("A.Id")\
    .agg(avg("TotalSteps").alias("AvgSteps"), avg("TotalMinutesAsleep").alias("AvgMinutesAsleep"))\
    #.withColumn("AvgSteps", round(col("AvgSteps"), 2))\
    #.withColumn("AvgMinutesAsleep", round(col("AvgMinutesAsleep"), 2))

#display(userSegments)

# Apply the conditions using 'when' function and create a new column 'UserSegment'
q8 = userSegments.withColumn("UserSegment",
    when((col("AvgSteps") >= 10000) & (col("AvgMinutesAsleep") >= 420), 'Active Sleepers')
    .when((col("AvgSteps") >= 10000) & (col("AvgMinutesAsleep") < 420),  'Active, Less Sleep')
    .when((col("AvgSteps") < 10000)  & (col("AvgMinutesAsleep") >= 420), 'Less Active, Good Sleep')
    .when((col("AvgSteps") < 10000)  & (col("AvgMinutesAsleep") < 420),  'Less Active, Less Sleep')
    .otherwise("Other"))

q8 = q8.select("Id", "AvgSteps", "AvgMinutesAsleep", "UserSegment")

display(q8)

# COMMAND ----------

# MAGIC %md
# MAGIC - User segmentation involves categorizing users based on certain characteristics or behaviors. In this case, we want to segment users based on their activity and sleep patterns.
# MAGIC - Less Active, Less Sleep: Users in this group have lower average steps, indicating a less active lifestyle.They also have a shorter average sleep duration (around 418 minutes).These users might benefit from interventions to increase physical activity and improve sleep habits.
# MAGIC - Active, Less Sleep:Users in this group are more active, as evidenced by a higher average step count.However, they still have a relatively shorter average sleep duration (around 418 minutes).Strategies to maintain activity levels while improving sleep quality could be explored for this group.
# MAGIC - Less Active, Good Sleep:This group has lower average steps but a longer and presumably better sleep duration (around 435 minutes).While these users are less active, they seem to prioritize and achieve better sleep.Understanding factors contributing to their good sleep could be valuable.
# MAGIC - These insights provide a high-level understanding of user behavior, allowing for targeted interventions or personalized recommendations.

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

q8Tmp = q8.toPandas()

# Map unique IDs to consecutive numbers
id_mapping = {id_: i+1 for i, id_ in enumerate(q8Tmp['Id'].unique())}

# Map IDs to consecutive numbers
q8Tmp['Id_Consecutive'] = q8Tmp['Id'].map(id_mapping)

y = q8Tmp.AvgSteps
x = q8Tmp.Id_Consecutive
z = q8Tmp.AvgMinutesAsleep

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

#set size
#plt.figure(figsize=(15, 3))
#plt.tight_layout()
  
# First subplot (index 0)
axs[0].bar(x, y)
axs[0].set_ylabel('Steps')
axs[0].set_xlabel('Id')
axs[0].legend()

# Second subplot (index 1)
axs[1].scatter(x, z, label="Sleep")
axs[1].set_ylabel('Minutes Asleep')
axs[1].set_xlabel('Id')
axs[1].legend()

#plt.xticks(rotation=90)
plt.ylabel('sleep')
plt.xlabel('Id')

plt.legend() 
plt.show()

# COMMAND ----------

#TODO: to fix => add data from the df

import matplotlib.pyplot as plt

# Prepare your data
labels = ['Active, Less Sleep','Less Active, Good Sleep','Less Active, Less Sleep']
sizes = [22, 3, 75]

fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

ax.axis('equal')  

plt.title('User Segments')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Sleep and Calories Comparison

# COMMAND ----------

from pyspark.sql.functions import sum
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

joined_df = dailyActivityFormatted.alias('A') \
  .join(dailySleepFormatted.alias('S'), (col('A.Id') == col('S.Id')) & (col('A.Date') == col('S.SleepDay')), 'inner')

q9 = joined_df\
    .groupBy("A.Id")\
    .agg(
      sum("TotalMinutesAsleep").alias("TotalMinutesAsleep"),
      sum("TotalTimeInBed").alias("TotalTimeInBed"),
      sum("Calories").alias("Calories")
    )

#display(q9)

q9Tmp = q9.toPandas()

x = q9Tmp.Calories
y = q9Tmp.TotalMinutesAsleep

# Fit a linear regression model
model = LinearRegression()
model.fit(x.values.reshape(-1, 1), y)

# Predict y values using the model
y_pred = model.predict(x.values.reshape(-1, 1))

#set size
plt.figure(figsize=(15, 3))
plt.tight_layout()

# Plot the scatterplot
plt.scatter(x, y, color='blue', alpha=0.5)

# Plot the line of best fit
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression')

plt.title('TotalMinutesAsleep vs Calories')
plt.xlabel('Calories')
plt.ylabel('Total Minutes Asleep')
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - The average sleep duration varies, indicating diverse sleep patterns among users.
# MAGIC - Users with higher activity levels or longer awake periods may tend to burn more calories.
# MAGIC - Some users have longer sleep durations but spend less time in bed, while others may have shorter sleep durations with more time in bed

# COMMAND ----------

# MAGIC %md
# MAGIC ##Average METs for single user

# COMMAND ----------

#display(minuteMETs)
#minuteMETs.printSchema

from pyspark.sql.functions import col, avg

minuteMETs = minuteMETsFormatted.drop("Minute")
grouped_df = minuteMETsFormatted.groupBy("Id","Date")

q10 = grouped_df.agg(avg(col("METs")).alias("avg_METs"))
q10 = q10.orderBy("Id","Date")
q10 = q10.filter("Id == '1624580081'")

display(q10)
