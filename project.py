# Databricks notebook source
# MAGIC %md #1) FitBit Project

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1) Description
# MAGIC This dataset includes health data about 30 people, such as daily activity, heart rate, sleep monitoring etc., tracked by their FitBit over a period of 30 days (from 12/04/2016 to 12/05/2016).
# MAGIC
# MAGIC Our goal is to analyze the athletes' routine to recognize patterns and trends in their activities.

# COMMAND ----------

# MAGIC %md #2) Import datasets

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

# MAGIC %md #3) Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.0) Libraries

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, dayofweek, date_format, round, udf, count, to_date, split,to_timestamp, avg, sum, when
from pyspark.sql.types import DateType
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import random

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.1) Daily Activity

# COMMAND ----------

def sumDistances(x,y,z,k):
	return x+y+z+k

sum_udf=udf(sumDistances,DoubleType())

# COMMAND ----------

dailyActivityFormatted = dailyActivity

dailyActivityFormatted = dailyActivityFormatted\
    .withColumn("TotalSteps", col("TotalSteps").cast("int"))\
    .withColumn("WeekDayN", dayofweek("ActivityDate"))\
    .withColumn("WeekDay", date_format(col("ActivityDate"), "E"))\
    .withColumnRenamed('ActivityDate','Date')    

# Creates new column: SumOfDistances
dailyActivityFormatted = dailyActivityFormatted.alias('A')\
    .select(\
        dailyActivityFormatted.Id,\
        dailyActivityFormatted.Date,\
        dailyActivityFormatted.TotalSteps,\
        dailyActivityFormatted.TotalDistance,\
        dailyActivityFormatted.SedentaryActiveDistance,\
        dailyActivityFormatted.LightActiveDistance,\
        dailyActivityFormatted.ModeratelyActiveDistance,\
        dailyActivityFormatted.VeryActiveDistance,\
        dailyActivityFormatted.SedentaryMinutes,\
        dailyActivityFormatted.LightlyActiveMinutes,\
        dailyActivityFormatted.FairlyActiveMinutes,\
        dailyActivityFormatted.VeryActiveMinutes,\
        dailyActivityFormatted.Calories,\
        dailyActivityFormatted.WeekDay,\
        dailyActivityFormatted.WeekDayN,\
        sum_udf(\
            col('A.VeryActiveDistance'),\
            col('A.ModeratelyActiveDistance'),\
            col('A.LightActiveDistance'),\
            col('A.SedentaryActiveDistance')))\
    .withColumnRenamed('sumDistances(VeryActiveDistance, ModeratelyActiveDistance, LightActiveDistance, SedentaryActiveDistance)','SumOfDistances')

# Remove rows with missing distance values, otherwise we will have fake outliers
dailyActivityFormatted = dailyActivityFormatted\
    .filter("SumOfDistances>0")\
    .orderBy('Date','Id')

display(dailyActivityFormatted)

# COMMAND ----------

# Check missing-values 
missing_count = dailyActivityFormatted\
    .filter(dailyActivityFormatted["Id"].isNull()).count()

print("Missing Count:", missing_count)

# Check duplicates
grouped_df = dailyActivityFormatted.groupBy("Id", "Date", "TotalSteps").agg(count("*").alias("Count"))

filtered_df = grouped_df.filter(grouped_df["Count"] > 1)
result_df = filtered_df.select("Id", "Date", "TotalSteps", "Count")

print("Duplicates:", result_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.2) Minute METs
# MAGIC MET (Metabolic Equivalent of Task) is a physiological measure expressing the energy cost of physical activities.<br>
# MAGIC MET values are often used to estimate calorie expenditure during physical activities.

# COMMAND ----------

minuteMETsFormatted = minuteMETs

minuteMETsFormatted=minuteMETsFormatted\
  .withColumnRenamed('Activity','Date')\
  .withColumnRenamed('Minute','Time')

display(minuteMETsFormatted)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.3) Daily Sleep

# COMMAND ----------

dailySleepFormatted = dailySleep

# Split the column 'SleepDay' in order to get just the date
dailySleepFormatted = dailySleepFormatted\
    .withColumn("SleepDay", split(dailySleep.SleepDay, " ")[0])

# Add the new columns 'WeekDay' and 'WeekDayN'
dailySleepFormatted = dailySleepFormatted\
    .withColumn("SleepDay", to_date(dailySleepFormatted.SleepDay, "M/d/yyyy").cast(DateType()))\
    .withColumn("WeekDayN", dayofweek("SleepDay"))\
    .withColumn("WeekDay", date_format(col("SleepDay"), "E"))\
    .drop('TotalSleepRecords')\
    .withColumnRenamed('SleepDay','Date')\
    .orderBy('Date','Id')

display(dailySleepFormatted)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.4) Hourly Steps

# COMMAND ----------

# Convert the string into a timestamp
hourlyStepsFormatted = hourlySteps\
    .withColumn("ActivityHour", to_timestamp("ActivityHour", "M/d/yyyy h:mm:ss a"))

# Format the timestamp into the desired format
hourlyStepsFormatted = hourlyStepsFormatted\
    .withColumn("ActivityHour", date_format("ActivityHour", "M/d/yyyy HH:mm"))

# Split the 'ActivityHour' column into two columns: 'Day' and 'Time'
hourlyStepsFormatted = hourlyStepsFormatted\
    .withColumn("Day", split(col("ActivityHour"), " ")[0])\
    .withColumn("Hour", split(col("ActivityHour"), " ")[1])

hourlyStepsFormatted = hourlyStepsFormatted\
    .withColumnRenamed('StepTotal','Steps')

# Reorder the columns as per your requirement
hourlyStepsFormatted = hourlyStepsFormatted\
    .select("Id", "Day", "Hour", "Steps")

display(hourlyStepsFormatted)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.5) Daily Intensity

# COMMAND ----------

dailyIntensitiesFormatted = dailyIntensities\
    .withColumnRenamed('ActivityDay','Date')\
    .orderBy('Id','Date')

display(dailyIntensitiesFormatted)
#display(dailyActivityFormatted)

#print("dailyActivity:",dailyActivityFormatted.count())
#print("dailyIntensity:",dailyIntensitiesFormatted.count())

#TODO: to fix 
tmp = dailyActivityFormatted.alias('A')\
    .join(dailyIntensitiesFormatted.alias('I'),("Date"), "inner")

#display(tmp.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.6) Other 

# COMMAND ----------

#Number of Users
users = dailyActivity.select("Id").distinct()
#display(users) # 33 users 

# COMMAND ----------

# MAGIC %md #4) Analyses

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.1) Daily Average Analysis
# MAGIC A daily average analysis based on the provided smart device usage data, we can calculate the average values for relevant metrics for each day

# COMMAND ----------

q1 = (
    dailyActivityFormatted
    .orderBy("WeekDayN","WeekDay")
    .groupBy("WeekDayN","WeekDay")
    .agg(avg("TotalSteps").alias("AvgSteps"),
         avg("TotalDistance").alias("AvgDistance"),
         avg("Calories").alias("AvgCalories"))
)

'''
q1 = q1\
    .withColumn("avg_steps", round(col("avg_steps"), 2))\
    .withColumn("avg_distance", round(col("avg_distance"), 2))\
    .withColumn("avg_calories", round(col("avg_calories"), 2))
'''

display(q1)

# COMMAND ----------

# MAGIC %md
# MAGIC Average Steps Per Day

# COMMAND ----------

q1Tmp = q1.toPandas()

x = q1Tmp.WeekDay
y1 = q1Tmp.AvgSteps
y2 = q1Tmp.AvgDistance
y3 = q1Tmp.AvgCalories

fig, ax = plt.subplots(figsize=(15, 5))

ax.bar(x, y1)
#ax.bar(x, y2)
#ax.bar(x, y3)

ax.set_xlabel('Days')
ax.set_ylabel('Steps')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC During which hour of the day were the more calories burned?

# COMMAND ----------

#TODO: make a chart

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.2) Compare Different Metrics
# MAGIC Duration of each Activity and Calories Burned Per User

# COMMAND ----------

q2 = (
    dailyActivityFormatted
    .groupBy("Id")
    .agg(
        sum("TotalSteps").alias("TotalSteps"),
        sum("LightlyActiveMinutes").alias("TotalLightlyActiveMinutes"),
        sum("FairlyActiveMinutes").alias("TotalFairlyActiveMinutes"),
        sum("VeryActiveMinutes").alias("TotalVeryActiveMinutes"),
        sum("Calories").alias("TotalCalories")
    )
)

display(q2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.3) Activity Time and Calories Burned

# COMMAND ----------

# Scatter plot between TotalCalories and TotalSteps
q2Tmp = q2.toPandas()

x = q2Tmp.TotalCalories
y = q2Tmp.TotalSteps

plt.scatter(x, y)

model = LinearRegression()
model.fit(x.values.reshape(-1, 1), y)

# Predict y values using the model
y_pred = model.predict(x.values.reshape(-1, 1))

# Plot the scatterplot
plt.scatter(x, y, color='blue', alpha=0.5)

# Plot the line of best fit
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression')

plt.title('TotalCalories VS TotalSteps')
plt.xlabel('TotalCalories')
plt.ylabel('TotalSteps')
plt.grid()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The total steps vary significantly across users, ranging from as low as 12,352 to as high as 702,840. This indicates diverse levels of physical activity among users.

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.4) Average Steps Per Hours

# COMMAND ----------

q3 = (
    hourlyStepsFormatted
    .groupBy("Hour")
    .agg(avg(col("Steps")).alias("avg_steps"))
    .orderBy("Hour")
)

#display(q3)

q3Tmp = q3.toPandas()

x = q3Tmp.Hour
y = q3Tmp.avg_steps

fig, ax = plt.subplots(figsize=(15, 5))

ax.bar(x, y)

ax.set_xlabel('Hours')
ax.set_ylabel('Steps')

plt.show()

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

# MAGIC %md ##3.5) Metrics Comparison Over Time
# MAGIC This query can help understand if there are consistent patterns or if certain metrics are more influential in different periods.

# COMMAND ----------

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
# MAGIC ##3.6) Categorize users into different intensity levels

# COMMAND ----------

q5 = (
    minuteMETsFormatted
    .groupBy("Id")
    .agg(avg("METs").alias("AvgMETs"))
    .withColumn("AvgMETs", round(col("AvgMETs")))
    .withColumn("IntensityLevel",
        when(col("AvgMETs") <= 10, "Low")
        .when((col("AvgMETs") > 10) & (col("AvgMETs") <= 16), "Medium")
        .when(col("AvgMETs") > 16, "High")
        .otherwise("n/d")
    )
)

#display(q5)

q5Tmp = q5\
    .groupBy('IntensityLevel')\
    .agg(sum("AvgMETs").alias("AvgMETs"),)

q5Tmp = q5Tmp.toPandas()

labels = q5Tmp['IntensityLevel']
sizes = q5Tmp['AvgMETs']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  
#plt.title('User Segments')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The output shows the average METs and the corresponding intensity category for each user.
# MAGIC
# MAGIC In this case, most users are categorized as "Medium" based on the provided threshold values.
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
# MAGIC ##3.7) Intensity of Activities
# MAGIC Exploring the distribution of METs to understand the range of activity intensities recorded by the devices.
# MAGIC Identifying peak MET values and correlating them with specific activities or time periods.

# COMMAND ----------

q6 = (
    minuteMETsFormatted
    .groupBy("METs")
    .count()
    .orderBy("METs")
)

#display(q6)

#TODO: add chart

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.8) Sleep analysys
# MAGIC Analyzing the average total minutes asleep to understand the typical sleep duration.<br>
# MAGIC Looking for trends or patterns in sleep data over time.

# COMMAND ----------

dailySleepFormatted2 = dailySleepFormatted

# Select necessary columns and calculate average TotalTimeInBed grouped by Weekday
sleepmean = dailySleepFormatted2\
    .select("WeekDayN","WeekDay","TotalMinutesAsleep")\
    .groupBy("WeekDayN","WeekDay")\
    .agg(round(avg("TotalMinutesAsleep")).alias("AvgSleepMinutes"))

sleepmean = sleepmean.orderBy("WeekDayN")

#display(sleepmean)

sleepmeanTmp = sleepmean.toPandas()

x = sleepmeanTmp.WeekDay
y = sleepmeanTmp.AvgSleepMinutes

fig, ax = plt.subplots(figsize=(15, 5))

ax.bar(x, y)

ax.set_xlabel('Days')
ax.set_ylabel('Average Minute Asleep')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see all the users have almost the same quantity of sleep during the week, which is around 6 hours and a half (with a little increse in Sunday and Wednesday)

# COMMAND ----------

# MAGIC %md
# MAGIC Exploring day-to-day variability in both physical activity and sleep metrics. Identifying trends or unusual events that might impact users' routines.

# COMMAND ----------

#Sleep Trend Analysis

joined_df = dailyActivityFormatted\
    .join(dailySleepFormatted, dailyActivityFormatted["Date"] == dailySleepFormatted["Date"], "inner")

q7 = joined_df\
    .select(dailySleepFormatted.Date, "TotalMinutesAsleep")\
    .orderBy(dailySleepFormatted.Date)

display(q7)

# COMMAND ----------

# MAGIC %md ##3.9) User classification
# MAGIC Segmenting users based on their activity and sleep patterns. This can help identify different user groups with distinct behaviors.

# COMMAND ----------

activityAndSleep = dailyActivityFormatted.alias('A')\
    .join(dailySleepFormatted.alias('S'), col("A.Date") == col("S.Date"), "inner")

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
plt.grid()
plt.legend() 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - User segmentation involves categorizing users based on certain characteristics or behaviors. In this case, we want to segment users based on their activity and sleep patterns.
# MAGIC - Less Active, Less Sleep: Users in this group have lower average steps, indicating a less active lifestyle.They also have a shorter average sleep duration (around 418 minutes).These users might benefit from interventions to increase physical activity and improve sleep habits.
# MAGIC - Active, Less Sleep:Users in this group are more active, as evidenced by a higher average step count.However, they still have a relatively shorter average sleep duration (around 418 minutes).Strategies to maintain activity levels while improving sleep quality could be explored for this group.
# MAGIC - Less Active, Good Sleep:This group has lower average steps but a longer and presumably better sleep duration (around 435 minutes).While these users are less active, they seem to prioritize and achieve better sleep.Understanding factors contributing to their good sleep could be valuable.
# MAGIC - These insights provide a high-level understanding of user behavior, allowing for targeted interventions or personalized recommendations.

# COMMAND ----------

q8Tmp = q8\
    .groupBy('UserSegment')\
    .agg(sum("AvgMinutesAsleep").alias("AvgMinutesAsleep"),)

q8Tmp = q8Tmp.toPandas()

labels = q8Tmp['UserSegment']
sizes = q8Tmp['AvgMinutesAsleep']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  
#plt.title('User Segments')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.10) Sleep and Calories Comparison

# COMMAND ----------

joined_df = dailyActivityFormatted.alias('A') \
  .join(dailySleepFormatted.alias('S'), (col('A.Id') == col('S.Id')) & (col('A.Date') == col('S.Date')), 'inner')

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
plt.figure(figsize=(15, 5))
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
# MAGIC ##3.11) Thomas analysys

# COMMAND ----------

df1 = dailyActivity.select("Id", "ActivityDate","Calories").\
    withColumnRenamed("ActivityDate", "DayA").\
    withColumnRenamed("Id", "IdA")
#display(df1)

df2 = dailyHeartrate.select("Id", "Day", "Mean_Value").\
    withColumnRenamed("Mean_Value", "AvgHeartrate")
#display(df2)

q11 = df1.join(df2,(df1["DayA"]==df2["Day"]) & (df1["IdA"]==df2["Id"]), "inner")\
    .drop("DayA", "IdA")

q11 = q11.select("Id", "Day", "Calories", "AvgHeartrate")
display(q11)

# COMMAND ----------

# Collect all distinct IDs
distinct_ids = q11.select("Id").distinct().collect()

selected_ids = []

# Get random IDs for each P
random.seed(12)
for _ in range(4):
    # Ensure the randomly selected ID is unique
    while True:
        random_id = random.choice(distinct_ids)[0]
        if random_id not in selected_ids:
            selected_ids.append(random_id)
            break

# Filter DataFrame to select data only where ID matches the randomly selected IDs
P1 = q11.filter(col("Id") == selected_ids[0]).alias("P1")
#display(P1)

P2 = q11.filter(col("Id") == selected_ids[1]).alias("P2")
#display(P2)

P3 = q11.filter(col("Id") == selected_ids[2]).alias("P3")
#display(P3)

P4 = q11.filter(col("Id") == selected_ids[3]).alias("P4")
#display(P4)

# COMMAND ----------

x1 = P1.select('AvgHeartrate').toPandas()['AvgHeartrate']
y1 = P1.select('Calories').toPandas()['Calories']

x2 = P2.select('AvgHeartrate').toPandas()['AvgHeartrate']
y2 = P2.select('Calories').toPandas()['Calories']

x3 = P3.select('AvgHeartrate').toPandas()['AvgHeartrate']
y3 = P3.select('Calories').toPandas()['Calories']

x4 = P4.select('AvgHeartrate').toPandas()['AvgHeartrate']
y4 = P4.select('Calories').toPandas()['Calories']

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

#set size
#plt.figure(figsize=(15, 3))
#plt.tight_layout()
  
# First subplot
axs[0, 0].scatter(x1, y1, label="Person 1")
axs[0, 0].set_ylabel('Calories')
axs[0, 0].set_xlabel('Heartrate')
axs[0, 0].legend()

model1 = LinearRegression()
model1.fit(x1.values.reshape(-1, 1), y1)

# Predict y values using the model
y_pred1 = model1.predict(x1.values.reshape(-1, 1))

# Plot the line of best fit
axs[0, 0].plot(x1, y_pred1, color='red', linewidth=2)

# Second subplot
axs[0, 1].scatter(x2, y2, label="Person 2")
axs[0, 1].set_ylabel('Calories')
axs[0, 1].set_xlabel('Heartrate')
axs[0, 1].legend()

model2 = LinearRegression()
model2.fit(x2.values.reshape(-1, 1), y2)

# Predict y values using the model
y_pred2 = model2.predict(x2.values.reshape(-1, 1))

# Plot the line of best fit
axs[0, 1].plot(x2, y_pred2, color='red', linewidth=2)

# Third subplot
axs[1, 0].scatter(x3, y3, label="Person 3")
axs[1, 0].set_ylabel('Calories')
axs[1, 0].set_xlabel('Heartrate')
axs[1, 0].legend()

model3 = LinearRegression()
model3.fit(x3.values.reshape(-1, 1), y3)

# Predict y values using the model
y_pred3 = model3.predict(x3.values.reshape(-1, 1))

# Plot the line of best fit
axs[1, 0].plot(x3, y_pred3, color='red', linewidth=2)

# Fourth subplot
axs[1, 1].scatter(x4, y4, label="Person 4")
axs[1, 1].set_ylabel('Calories')
axs[1, 1].set_xlabel('Heartrate')
axs[1, 1].legend()

model4 = LinearRegression()
model4.fit(x4.values.reshape(-1, 1), y4)

# Predict y values using the model
y_pred4 = model4.predict(x4.values.reshape(-1, 1))

# Plot the line of best fit
axs[1, 1].plot(x4, y_pred4, color='red', linewidth=2)

#plt.xticks(rotation=90)

plt.legend() 
plt.show()

# COMMAND ----------

P3_tmp = P3.join(dailyActivity, (P3["Id"]==dailyActivity["Id"]) & (P3["Day"]==dailyActivity["ActivityDate"]))
#display(P3_tmp)
P3_1 = P3_tmp.select(P3["Id"], "Day", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes", P3["Calories"])
display(P3_1)

# COMMAND ----------

# MAGIC %md
# MAGIC While high heartrate during exercise has a positive correlation with calories burned in MOST individuals, we can see that there are outliers such as P3. They burned more calories during low intensity exercise.

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.12) Average METs for single user (?)

# COMMAND ----------

#display(minuteMETs)

minuteMETs = minuteMETsFormatted.drop("Minute")
grouped_df = minuteMETsFormatted.groupBy("Id","Date")

q10 = grouped_df.agg(avg(col("METs")).alias("AvgMETs"))
q10 = q10.orderBy("Id","Date")
q10 = q10.filter("Id == '1624580081'")

display(q10)

# COMMAND ----------

# MAGIC %md
# MAGIC #5) Conclusion
# MAGIC
# MAGIC In this comprehensive analysis of Bellabeat's smart device usage data, we delved into various aspects of user behavior, ranging from daily activity patterns to sleep metrics. The analysis aimed to provide actionable insights for Bellabeat's marketing strategy by understanding trends and identifying potential opportunities for growth.
# MAGIC
# MAGIC Here are the key findings and recommendations:
# MAGIC
# MAGIC Daily Activity Patterns:
# MAGIC Activity Distribution Over the Day: Users tend to be more active during the morning and early afternoon, with a peak in steps between 8:00 AM and 7:00 PM.Understanding peak activity hours allows for targeted engagement, promotions, or notifications during these times.
# MAGIC
# MAGIC Metrics Comparison Over Time: The distribution of metrics such as steps, distance, and calories burned varies over time.Identifying trends or unusual events in these metrics can help understand users routines and tailor marketing strategies accordingly.
# MAGIC
# MAGIC Intensity of Activities:
# MAGIC Categorization by Intensity:Users were categorized into intensity levels based on average METs.All users were classified as engaging in "Vigorous Intensity" activities.Tailoring recommendations, features, and content for users involved in vigorous activities could enhance engagement.
# MAGIC
# MAGIC METs Distribution:Exploring the distribution of METs provided insights into the range of activity intensities recorded by the devices.Bellabeat can leverage METs data to categorize activities and offer personalized recommendations for users.
# MAGIC
# MAGIC Sleep Patterns:
# MAGIC Identifying Sleep Patterns:Analyzing sleep data revealed average sleep durations for each day of the week.Understanding day-to-day variability in sleep patterns can help tailor wellness features or recommendations.
# MAGIC
# MAGIC User Segmentation:Users were segmented based on their activity and sleep patterns.Segments include "Less Active, Less Sleep," "Active, Less Sleep," and more, providing insights for targeted interventions.
# MAGIC
# MAGIC Sleep and Calories Comparison:Examining the relationship between sleep metrics and calories burned highlighted variations in sleep duration and calories expended during activities.
# MAGIC
# MAGIC Recommendations:
# MAGIC Tailor marketing strategies and product features based on the diverse user profiles identified in the analysis.
# MAGIC Provide specialized content, challenges, or workouts for users with specific activity patterns or intensity levels.
# MAGIC Use insights into peak activity hours to optimize engagement strategies, promotions, or notifications during high-activity periods.
# MAGIC Leverage insights into sleep patterns to enhance sleep-related features or offer personalized recommendations for better sleep.
# MAGIC Foster a sense of community among users with similar activity levels or goals through forums, groups, or challenges.
# MAGIC Ensure seamless integration with wearables during vigorous activities to capture accurate and real-time data.

# COMMAND ----------

# MAGIC %md
# MAGIC #6) Reference (to delete)
# MAGIC https://www.kaggle.com/code/deepalisukhdeve/data-driven-wellness#Analyze
