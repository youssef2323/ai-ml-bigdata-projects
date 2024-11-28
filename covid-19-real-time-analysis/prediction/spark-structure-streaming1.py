from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StringType, IntegerType , FloatType
import pymysql

#conn = pymysql.connect(host=host, port=port, user=username, passwd=password, db=database)
#cursor = conn.cursor()

def insert_into_phpmyadmin(row):
    # Define the connection details for your PHPMyAdmin database
    host = "localhost"
    port = 3306
    database = "big_data"
    username = "root"
    password = ""

    conn = pymysql.connect(host=host, port=port, user=username, passwd=password, db=database)
    cursor = conn.cursor()

    # Extract the required columns from the row

    column1_value = row.total_deaths
    column2_value = row.prediction

    # Prepare the SQL query to insert data into the table
    sql_query = f"INSERT INTO ft(`total_deaths`, `prediction`) VALUES ('{column1_value}', '{column2_value}')"

    # Execute the SQL query
    cursor.execute(sql_query)

    # Commit the changes
    conn.commit()
    conn.close()

# Create a Spark session
spark = SparkSession.builder \
    .appName("KafkaConsumer") \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')

# Define the schema for your DataFrame
schema = (
    StructType()
    .add("iso_code", StringType())
    .add("continent", StringType())
    .add("location", StringType())
    .add("date", StringType())
    .add("total_cases", FloatType())
    .add("new_cases", FloatType())
    .add("new_cases_smoothed", FloatType())
    .add("total_deaths", FloatType())
    .add("new_deaths", FloatType())
    .add("new_deaths_smoothed", FloatType())
    .add("total_cases_per_million", FloatType())
    .add("new_cases_per_million", FloatType())
    .add("new_cases_smoothed_per_million", FloatType())
    .add("total_deaths_per_million", FloatType())
    .add("new_deaths_per_million", FloatType())
    .add("new_deaths_smoothed_per_million", FloatType())
    .add("reproduction_rate", FloatType())
    .add("icu_patients", FloatType())
    .add("icu_patients_per_million", FloatType())
    .add("hosp_patients", FloatType())
    .add("hosp_patients_per_million", FloatType())
    .add("weekly_icu_admissions", FloatType())
    .add("weekly_icu_admissions_per_million", FloatType())
    .add("weekly_hosp_admissions", FloatType())
    .add("weekly_hosp_admissions_per_million", FloatType())
    .add("total_tests", FloatType())
    .add("new_tests", FloatType())
    .add("total_tests_per_thousand", FloatType())
    .add("new_tests_per_thousand", FloatType())
    .add("new_tests_smoothed", FloatType())
    .add("new_tests_smoothed_per_thousand", FloatType())
    .add("positive_rate", FloatType())
    .add("tests_per_case", FloatType())
    .add("tests_units", FloatType())
    .add("total_vaccinations", FloatType())
    .add("people_vaccinated", FloatType())
    .add("people_fully_vaccinated", FloatType())
    .add("total_boosters", FloatType())
    .add("new_vaccinations", FloatType())
    .add("new_vaccinations_smoothed", FloatType())
    .add("total_vaccinations_per_hundred", FloatType())
    .add("people_vaccinated_per_hundred", FloatType())
    .add("people_fully_vaccinated_per_hundred", FloatType())
    .add("total_boosters_per_hundred", FloatType())
    .add("new_vaccinations_smoothed_per_million", FloatType())
    .add("new_people_vaccinated_smoothed", FloatType())
    .add("new_people_vaccinated_smoothed_per_hundred", FloatType())
    .add("stringency_index", FloatType())
    .add("population_density", FloatType())
    .add("extreme_poverty", FloatType())
    .add("cardiovasc_death_rate", FloatType())
    .add("diabetes_prevalence", FloatType())
    .add("handwashing_facilities", FloatType())
    .add("hospital_beds_per_thousand", FloatType())
    .add("life_expectancy", FloatType())
    .add("human_development_index", FloatType())
    .add("population", IntegerType())
    .add("excess_mortality_cumulative_absolute", FloatType())
    .add("excess_mortality_cumulative", FloatType())
    .add("excess_mortality", FloatType())
    .add("excess_mortality_cumulative_per_million", FloatType())
)

schema2 = StructType().add("total_deaths",FloatType()).add("prediction",FloatType())

# Read data from Kafka topic as a DataFrame
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "dull") \
    .load() \
    .select(from_json(col("value").cast("string"), schema2).alias("data")) \

df = df.select("data.total_deaths","data.prediction")

query = df.writeStream \
    .outputMode("append") \
    .format("console") \
    .foreach(insert_into_phpmyadmin) \
    .start()

# Wait for the query to finish
query.awaitTermination()
