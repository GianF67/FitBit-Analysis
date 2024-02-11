# Databricks notebook source
# MAGIC %md #1) FitBit Project

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1) Description
# MAGIC This dataset includes health data about 30 people, such as daily activity, heart rate, sleep monitoring etc., tracked by their FitBit devices over a period of 30 days (from 12/04/2016 to 12/05/2016).
# MAGIC
# MAGIC Our goal is to analyze the athletes' routine in order to recognize patterns and trends in their activities.

# COMMAND ----------

# MAGIC %md
# MAGIC ##1.2) Dateset
# MAGIC https://www.kaggle.com/datasets/arashnic/fitbit/data

# COMMAND ----------

# MAGIC %md
# MAGIC ##1.3) GitHub
# MAGIC https://github.com/GianF67/cloud-technologies-project.git

# COMMAND ----------

# MAGIC %md #2) Import datasets

# COMMAND ----------

# IMPORTANT: Before run all these commands make sure that you have imported all necessary .csv files as DBFS (check the ds directory)

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

hourlyCalories = spark.read \
    .format("csv") \
    .option("inferSchema", inferSchema) \
    .option("header", "true") \
    .option("sep", ",") \
    .load("/FileStore/tables/hourlyCalories.csv")

# COMMAND ----------

# MAGIC %md #3) Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.0) Import libraries

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, dayofweek, date_format, round, udf, count, to_date, split,to_timestamp, avg, sum, when, hour
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

'''
- Cast TotalSteps as int,
- Create two new columns 'WeekDay' and 'WeekDayN' as date
- Cast ActivityDate as date and then rename it as Date
'''
dailyActivityFormatted = dailyActivityFormatted\
    .withColumn("TotalSteps", col("TotalSteps").cast("int"))\
    .withColumn("WeekDayN", dayofweek("ActivityDate"))\
    .withColumn("WeekDay", date_format(col("ActivityDate"), "E"))\
    .withColumnRenamed('ActivityDate','Date')    

# Create a new column 'SumOfDistances'
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

# Add two new columns 'WeekDay' and 'WeekDayN'
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

hourlyStepsFormatted = hourlySteps

# Cast ActivityHour from string to timestamp
hourlyStepsFormatted = hourlyStepsFormatted\
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
# MAGIC ##3.5) Hourly Calories

# COMMAND ----------

# Convert the string into a timestamp
hourlyCaloriesFormatted = hourlyCalories\
    .withColumn("ActivityHour", to_timestamp("ActivityHour", "M/d/yyyy h:mm:ss a"))

# Format the timestamp into the desired format
hourlyCaloriesFormatted = hourlyCaloriesFormatted\
    .withColumn("ActivityHour", date_format("ActivityHour", "M/d/yyyy HH:mm"))

# Split the 'ActivityHour' column into two columns: 'Day' and 'Time'
hourlyCaloriesFormatted = hourlyCaloriesFormatted\
    .withColumn("Day", split(col("ActivityHour"), " ")[0])\
    .withColumn("Hour", split(col("ActivityHour"), " ")[1])

# Delete unecessary columns
hourlyCaloriesFormatted = hourlyCaloriesFormatted.drop('ActivityHour')

# Select relevant columns
hourlyCaloriesFormatted = hourlyCaloriesFormatted.select('Id','Day','Hour','Calories')

display(hourlyCaloriesFormatted)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.6) Daily Intensity

# COMMAND ----------

dailyIntensitiesFormatted = dailyIntensities

dailyIntensitiesFormatted = dailyIntensitiesFormatted\
    .withColumnRenamed('ActivityDay','Date')\
    .orderBy('Id','Date')

display(dailyIntensitiesFormatted)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.7) Other 

# COMMAND ----------

#Number of Users
users = dailyActivity.select("Id").distinct()
#display(users) # 33 users 

# COMMAND ----------

# MAGIC %md #4) Analyses

# COMMAND ----------

# Global variables

color_blu = '#205bc9'
color_red = '#eb4034'
color_green = '#0c661d'

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.1) Daily Average Analysis
# MAGIC With this analysis we want to calculate, for each day of the week, the average values for the main metrics:
# MAGIC - average steps
# MAGIC - average distance
# MAGIC - average calories (burned)

# COMMAND ----------

q1 = (
    dailyActivityFormatted
    .orderBy("WeekDayN","WeekDay")
    .groupBy("WeekDayN","WeekDay")
    .agg(avg("TotalSteps").alias("AvgSteps"),
         avg("TotalDistance").alias("AvgDistance"),
         avg("Calories").alias("AvgCalories"))
)

display(q1)

# COMMAND ----------

q1Tmp2 = q1.toPandas()

x = q1Tmp2.WeekDay
y1 = q1Tmp2.AvgSteps
y2 = q1Tmp2.AvgDistance
y3 = q1Tmp2.AvgCalories

bar_width = 0.25

# Set the position of each group of bars on the x-axis
r1 = np.arange(len(x))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Set the figure size
plt.figure(figsize=(15, 5))

# Plot the bars
plt.bar(r1, y1, color=color_blu, width=bar_width, label='AvgSteps')
#plt.bar(r2, y2, color='g', width=bar_width, edgecolor='grey', label='AvgDistance')
plt.bar(r3, y3, color=color_red, width=bar_width, label='AvgCalories')

# Set labels and title
plt.xlabel('WeekDay', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(x))], x)
plt.ylabel('Values', fontweight='bold')
plt.title('Daily Average Analysys')

plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see from the above chart the average steps is almost the same per each day, as for the amount of calories burned.

# COMMAND ----------

# MAGIC %md
# MAGIC Another interesting analysis is to understand during which hour of the day users burn more calories, providing insights into user engagement with their devices over time.

# COMMAND ----------

hourlyCaloriesFormatted2 = hourlyCaloriesFormatted\
  .groupBy(hour("Hour"))\
  .agg({"Calories": "avg"})\
  .withColumnRenamed("avg(Calories)", "AvgCalories")\
  .withColumnRenamed("hour(Hour)", "Hour")\
  .orderBy('Hour')

#display(hourlyCaloriesFormatted2)

hourlyCaloriesFormattedTmp = hourlyCaloriesFormatted2.toPandas()

hours = hourlyCaloriesFormattedTmp.Hour
avg_calories = hourlyCaloriesFormattedTmp.AvgCalories

plt.figure(figsize=(15, 5))

plt.bar(hours, avg_calories, color=color_blu)

plt.xlabel('Hours')
plt.ylabel('Average Calories')
plt.title('Average Calories Burned by Hour')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see from the chart, the time people want to be active during the day is between 8:00 and 19:00, probably because before 8:00 most people sleep and after 7:00 they prefer to stay at home.

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.2) Different Metrics Comparison
# MAGIC With this analysis we want to calculate, for each users, the duration of each activity with its relative burned calories.

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
# MAGIC The total steps vary significantly across users, within a range from 12000 to 500000, indicating diverse levels of physical activity among users.<br><br>
# MAGIC The following chart shows how as the steps increase the number of calories burned also increases.

# COMMAND ----------

q2Tmp = q2.toPandas()

x = q2Tmp.TotalCalories
y = q2Tmp.TotalSteps

plt.scatter(x, y)

model = LinearRegression()
model.fit(x.values.reshape(-1, 1), y)

# Predict y values using the model
y_pred = model.predict(x.values.reshape(-1, 1))

# Plot the scatterplot
plt.scatter(x, y, color=color_blu, alpha=0.5)

# Plot the line of best fit
plt.plot(x, y_pred, color=color_red, linewidth=2, label='Regression')

plt.title('TotalCalories VS TotalSteps')
plt.xlabel('TotalCalories')
plt.ylabel('TotalSteps')

plt.grid()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.3) Average Steps Per Hours
# MAGIC With this analysis we want to calculate the average steps per hours. We noticed that the average number of steps varies throughout different time intervals of the day, revealing distinct patterns in user activity. <br><br>
# MAGIC As we can see from the chart, we can identify three intervals:
# MAGIC 1. Between midnight and 6, there is a decline in the average steps, implying lower activity levels, potentially during sleep.
# MAGIC 2. Between 6 to 19, indicating heightened activity during the morning hours, likely due to activities like morning walks or commutes, which indicates high level of activities.
# MAGIC 3. After 19 PM, there is decline in average steps, indicating decreased activity during nighttime hours.

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
# MAGIC ##4.4) Metrics Comparison Over Time
# MAGIC With this analysis we want to understand if there are patterns or if certain metrics are more influential in different periods of the months.

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
plt.figure(figsize=(15, 5))
plt.tight_layout()
  
graph1 = plt.plot(x, y1, label = "Steps",color=color_blu)
graph2 = plt.plot(x, y2, label = "Distance",color=color_red)
graph3 = plt.plot(x, y3, label = "Calories",color=color_green)
plt.xticks(rotation=90)
ax = plt.gca()

#set x tick density
ax.set_xticks( [* range(int(ax.get_xticks()[0])-1, int(ax.get_xticks()[-1]), int( (ax.get_xticks()[-1] - ax.get_xticks()[0])/(len(ax.get_xticks())-1) / 3 )) ] )

#set y tick density
ax.set_yticks( [* range(int(ax.get_yticks()[0]), int(ax.get_yticks()[-1])+1, int( (ax.get_yticks()[-1] - ax.get_yticks()[0])/(len(ax.get_yticks())-1) / 2 )) ] )

ax.set_xlabel('Values')
ax.set_ylabel('Days')

plt.legend() 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.5) Users Classification By Intensity Level
# MAGIC
# MAGIC In this analysis we want to classify users based on their activity level, by analyzing the intensity of physical activities and correlate it with other metrics like steps, distance, and calories burned. To do so we are using the dataframe minuteMETsFormatted.
# MAGIC
# MAGIC First of all, let's define what are METs or Metabolic Equivalent of Task?
# MAGIC - METs are a measure of the energy expenditure of physical activities.
# MAGIC - One MET is defined as the energy expenditure at rest, which is equivalent of sitting.
# MAGIC - In the context of health and fitness tracking, METs are valuable because they provide a standardized way to measure and compare the intensity of different physical activities. Understanding METs allows to categorize activities based on their energy expenditure, which is our goal.
# MAGIC
# MAGIC Here's how METs we categorized:
# MAGIC - Low Intensity (1-10 METs): Activities such as sitting, standing, or casual walking.
# MAGIC - Medium Intensity (11-15 METs): Activities like walking/cycling at a moderate pace, or light housework.
# MAGIC - High Intensity (16+ METs): Activities such as running/cycling at a high speed, or intense exercise (basically all ctivities that significantly raise your heart rate and breathing).

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

display(q5)

# COMMAND ----------

# MAGIC %md
# MAGIC The above table shows the average METs and the corresponding intensity level for each user in our dataset.
# MAGIC
# MAGIC As we can see from the below chart, most users are categorized as "Medium Intensity", which means that most of the physical activities they did required a moderate amount of energy expenditure.

# COMMAND ----------

q5Tmp = q5\
    .groupBy('IntensityLevel')\
    .agg(sum("AvgMETs").alias("AvgMETs"),)

q5Tmp = q5Tmp.toPandas()

labels = q5Tmp['IntensityLevel']
sizes = q5Tmp['AvgMETs']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  
plt.title('User Classification by intensity level')

plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##4.6) Intensity of Activities
# MAGIC In this analysy we are exploring the distribution of METs to understand the range of activity intensities recorded by the FitBit, and identifying peak MET values in order to correlate them with specific activities or time periods.

# COMMAND ----------

q6 = (
    minuteMETsFormatted
    .groupBy("METs")
    .count()
    .orderBy("METs")
    .select("METs", col("count").alias("Frequency"))
)

display(q6)

# COMMAND ----------

# MAGIC %md
# MAGIC The table above tells us how the frequency of METs is distributed in the data set, while the next graph is just a graphical representation of it.<br>As we can see that the greatest majority of the activities performed by the users have the same intensity, which correspond to a MET around 10 which corresponds to activities like light walking or light cycling.

# COMMAND ----------

q6Tmp = q6.toPandas()

'''
#barchart
x = q6Tmp.METs
y = q6Tmp.Frequency
fig, ax = plt.subplots()
ax.bar(x, y)
ax.set_xlabel('Frequency')
ax.set_ylabel('Count of METs')
'''

'''
boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(q6Tmp['Frequency'], vert=True)
plt.xlabel('Frequency')
plt.title('Frequency Distributiom')
'''

x = q6Tmp.METs
y = q6Tmp.Frequency

# Fit a linear regression model
model = LinearRegression()
model.fit(x.values.reshape(-1, 1), y)

# Predict y values using the model
y_pred = model.predict(x.values.reshape(-1, 1))

#set size
plt.figure(figsize=(15, 5))
plt.tight_layout()

# Plot the scatterplot
plt.scatter(x, y, color=color_blu, alpha=0.5)

# Plot the line of best fit
plt.plot(x, y_pred, color=color_red, linewidth=2, label='Regression')

plt.title('METs Frequency Diatribution')
plt.xlabel('METs')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.7) Sleep Analysys
# MAGIC Through analysis of the mean total minutes asleep, we gain insights into the typical duration and sleep pattern of our user over time.

# COMMAND ----------

dailySleepFormatted2 = dailySleepFormatted

# Select necessary columns and calculate average TotalTimeInBed grouped by Weekday
sleepmean = dailySleepFormatted2\
    .select("WeekDayN","WeekDay","TotalMinutesAsleep")\
    .groupBy("WeekDayN","WeekDay")\
    .agg(round(avg("TotalMinutesAsleep")).alias("AvgSleepMinutes"))

sleepmean = sleepmean.orderBy("WeekDayN")

#display(sleepmean)
sleepmean.printSchema()

sleepmeanTmp = sleepmean.toPandas()

x = sleepmeanTmp.WeekDay
y = sleepmeanTmp.AvgSleepMinutes

fig, ax = plt.subplots(figsize=(15, 5))

ax.bar(x, y)

plt.title('Daily sleep analysys')
ax.set_xlabel('Days')
ax.set_ylabel('Average Minute Asleep')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see all the users have almost the same quantity of sleep during the week, which is around 6 hours and a half (with a little increse in Sunday and Wednesday)

# COMMAND ----------

# MAGIC %md
# MAGIC This analysy explores the day-to-day variability in both physical activity and sleep metrics, with the aim to identify trends or unusual events that might impact users' routines.

# COMMAND ----------

joined_df = dailyActivityFormatted\
    .join(dailySleepFormatted, dailyActivityFormatted["Date"] == dailySleepFormatted["Date"], "inner")

q7 = joined_df\
    .select(dailySleepFormatted.Date, "TotalMinutesAsleep")\
    .orderBy(dailySleepFormatted.Date)

display(q7)

# COMMAND ----------

# MAGIC %md 
# MAGIC ##4.8) User Classification By Activities And Sleep
# MAGIC After having analyzed users daily activities and daily sleep routines, we want to segment them based on these two metrics (activity and sleep), with the aim of identify different user groups with distinct behaviors.
# MAGIC
# MAGIC Here how's we classfy user by their activiy and sleep routines:
# MAGIC - More Active, More Sleep: Users in this group are both very active and good sleepers, as evidenced by a higher average step count and high minutes in bed.
# MAGIC - More Active, Less Sleep: Users in this group are more active, as evidenced by a higher average step count. However, they still have a relatively shorter average sleep duration.
# MAGIC - Less Active, Less Sleep: Users in this group have lower average steps, indicating a less active lifestyle. They also have a shorter average sleep duration.
# MAGIC - Less Active, More Sleep: This group has lower average steps but a longer and presumably better sleep duration. While these users are less active, they seem to prioritize and achieve better sleep.

# COMMAND ----------

activityAndSleep = dailyActivityFormatted.alias('A')\
    .join(dailySleepFormatted.alias('S'), col("A.Date") == col("S.Date"), "inner")

# Group by Id and calculate average TotalSteps and TotalMinutesAsleep
userSegments = activityAndSleep\
    .groupBy("A.Id")\
    .agg(avg("TotalSteps").alias("AvgSteps"), avg("TotalMinutesAsleep").alias("AvgMinutesAsleep"))

#display(userSegments)

q8 = userSegments.withColumn("UserSegment",
    when((col("AvgSteps")  >= 10000) & (col("AvgMinutesAsleep") >= 420), 'More Active, More Sleep')
    .when((col("AvgSteps") >= 10000) & (col("AvgMinutesAsleep") < 420),  'More Active, Less Sleep')
    .when((col("AvgSteps") < 10000)  & (col("AvgMinutesAsleep") >= 420), 'Less Active, More Sleep')
    .when((col("AvgSteps") < 10000)  & (col("AvgMinutesAsleep") < 420),  'Less Active, Less Sleep')
    .otherwise("Other"))

q8 = q8.select("Id", "AvgSteps", "AvgMinutesAsleep", "UserSegment")

display(q8)

# COMMAND ----------

# MAGIC %md
# MAGIC The next charts are representation of the activities and sleep routines.

# COMMAND ----------

q8Tmp = q8.toPandas()

# Map unique IDs to consecutive numbers
id_mapping = {id_: i+1 for i, id_ in enumerate(q8Tmp['Id'].unique())}

# Map IDs to consecutive numbers
q8Tmp['Id_Consecutive'] = q8Tmp['Id'].map(id_mapping)

y = q8Tmp.AvgSteps
x = q8Tmp.Id_Consecutive
z = q8Tmp.AvgMinutesAsleep

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

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
plt.title('Activities and sleep analysys')
plt.ylabel('sleep')
plt.xlabel('Id')
plt.grid()
plt.legend() 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The next is the representation of the users classification based on activities and sleep metrics.

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
plt.title('User classification by activities and sleep routines')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC *Note that from the data we weren't able to find the segment "More Active, More Sleep"

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.9) Sleep & Calories Comparison
# MAGIC With this analysis we want to understand how sleep and calories burned are correlated each other.

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

display(q9)

# COMMAND ----------

# MAGIC %md
# MAGIC If we take a look at the table we can see that some users have shorter sleep durations (TotalMinutesAsleep) but they spend more time in bed (TotalMinuteInBed), which may indicates sleep disorder or troubles in falling asleep.

# COMMAND ----------

# MAGIC %md
# MAGIC The chart shows us that the average sleep duration varies, which indicates diverse sleep patterns among users. For example, we can see that for users with higher activity levels or longer awake periods may tend to burn more calories.

# COMMAND ----------

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
plt.scatter(x, y, color=color_blu, alpha=0.5)

# Plot the line of best fit
plt.plot(x, y_pred, color=color_red, linewidth=2, label='Regression')

plt.title('Total Minutes Asleep vs Calories Burned')
plt.xlabel('Calories')
plt.ylabel('Total Minutes Asleep')
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.10) Heartrate & Calories
# MAGIC This analysis is aimed at finding a correlation between heartrate and calorie consumption.

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

# MAGIC %md
# MAGIC As a sample on the data, we will now focus on four random users

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

# MAGIC %md
# MAGIC We can see from the graphs that while all users present a positive correlation between calorie consumption and a high heartrate, one of our sample users does not.
# MAGIC In fact, it seems they consumed more calories while mainting a lower heartrate.
# MAGIC Let's now see what kind of exercise this user does.

# COMMAND ----------

P3_tmp = P3.join(dailyActivity, (P3["Id"]==dailyActivity["Id"]) & (P3["Day"]==dailyActivity["ActivityDate"]))
#display(P3_tmp)
P3_1 = P3_tmp.select(P3["Id"], "Day", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes", P3["Calories"])
display(P3_1)

# COMMAND ----------

# MAGIC %md
# MAGIC By comparing the values in the FairlyActiveMinutes to those in the LightlyActiveMinutes and SedentaryMinutes, we can deduce that this user prefers low intensity exercise and that they burn the most calories doing low intensity activities. This suggests that not all individuals respond the same way to the same workout.

# COMMAND ----------

# MAGIC %md
# MAGIC #5) Conclusion
# MAGIC
# MAGIC In this thorough analysis of FitBit usage data, we delved into various aspects of user behavior, ranging from daily activity patterns to sleep metrics. The analysis aimed to look for trends and identifying potential pattern recognition among users. Here's the patterns we discovered:
# MAGIC
# MAGIC - Daily Activity:
# MAGIC     - Users tend to be more active during the morning and early afternoon, with a peak in steps between 8:00 AM and 7:00 PM.
# MAGIC     - The distribution of the main metrics such as steps, distance, and calories burned varies over time.
# MAGIC
# MAGIC - Users Classification by Intensity Level:
# MAGIC     - Users were classified into intensity levels based on the average METs, finding as "Medium Intensity" the most popular among the users.
# MAGIC     - METs distribution provided insights into the range of activity intensities recorded by the devices.
# MAGIC
# MAGIC - Users Classification by Activites And Sleep Routines:
# MAGIC     - Users were classified based on their activity and sleep routines. The segments include:
# MAGIC         - "Less Active, Less Sleep" (most popular among our users), 
# MAGIC         - "Less Active, More Sleep", 
# MAGIC         - "More Active, More Sleep"
# MAGIC         - "More Active, Less Sleep", 
# MAGIC     - Analysis of the correlation between sleep metrics and calories burned pointed the variations in sleep duration and calorie expenditure during activities.
# MAGIC
# MAGIC - Heartrate and Calories: 
# MAGIC     - In some individuals, high heartrate does not always mean higher calorie consumption. We should strive to recommend tailored training for each user, based on their own metabolism, to maximize calorie expenditure.
