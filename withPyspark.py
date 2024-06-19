from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import time
import matplotlib.pyplot as plt

# Inisialisasi SparkSession
spark = SparkSession.builder.appName("TextAnalysisWithSpark").getOrCreate()

# Memulai waktu eksekusi
start_time = time.time()

# Load dataset
directory = 'hasil-crawling'
df = spark.read.csv(f'{directory}/*.csv', header=True, inferSchema=True)

# Logging waktu untuk load dataset
load_dataset_time = time.time() - start_time
print(f"Time taken to load dataset: {load_dataset_time} seconds")

# Preprocessing menggunakan UDF di Spark
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
import re
from deep_translator import GoogleTranslator

# Inisialisasi translator
translator = GoogleTranslator(source='auto', target='en')

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Translate
    text = translator.translate(text)
    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    # Stemming
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Register UDF untuk Spark DataFrame
clean_text_udf = udf(lambda x: clean_text(x), StringType())
df = df.withColumn('tweet_clean', clean_text_udf(col('full_text')))

# Logging waktu untuk preprocessing
preprocessing_time = time.time() - start_time - load_dataset_time
print(f"Time taken for preprocessing: {preprocessing_time} seconds")

# Tokenisasi teks
tokenizer = Tokenizer(inputCol="tweet_clean", outputCol="words")
wordsData = tokenizer.transform(df)

# Menghitung TF (Term Frequency)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# Menghitung IDF (Inverse Document Frequency)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Logging waktu untuk TF-IDF
tfidf_time = time.time() - start_time - load_dataset_time - preprocessing_time
print(f"Time taken for TF-IDF computation: {tfidf_time} seconds")

# Sentimen Analysis menggunakan VADER Sentiment di Spark
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Inisialisasi VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Fungsi untuk menentukan label sentimen
def vader_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Register UDF untuk Spark DataFrame
vader_sentiment_udf = udf(lambda x: vader_sentiment(x), StringType())
rescaledData = rescaledData.withColumn("sentiment", vader_sentiment_udf(col("tweet_clean")))

# Logging waktu untuk sentiment analysis
sentiment_time = time.time() - start_time - load_dataset_time - preprocessing_time - tfidf_time
print(f"Time taken for sentiment analysis: {sentiment_time} seconds")

# Split data menjadi training dan testing set
(trainingData, testData) = rescaledData.randomSplit([0.8, 0.2], seed=42)

# Train Naive Bayes model
nb = NaiveBayes(featuresCol='features', labelCol='label', modelType='multinomial')

# Logging waktu untuk training Naive Bayes
start_train_nb_time = time.time()

nbModel = nb.fit(trainingData)

# Logging waktu untuk training Naive Bayes
train_nb_time = time.time() - start_train_nb_time
print(f"Time taken for training Naive Bayes: {train_nb_time} seconds")

# Prediksi menggunakan model Naive Bayes
nbPredictions = nbModel.transform(testData)

# Evaluasi model Naive Bayes
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
nbAccuracy = evaluator.evaluate(nbPredictions)
print(f"Naive Bayes - Accuracy: {nbAccuracy}")

# Train SVM model
svm = LinearSVC(featuresCol='features', labelCol='label')

# Logging waktu untuk training SVM
start_train_svm_time = time.time()

svmModel = svm.fit(trainingData)

# Logging waktu untuk training SVM
train_svm_time = time.time() - start_train_svm_time
print(f"Time taken for training SVM: {train_svm_time} seconds")

# Prediksi menggunakan model SVM
svmPredictions = svmModel.transform(testData)

# Evaluasi model SVM
svmAccuracy = evaluator.evaluate(svmPredictions)
print(f"SVM - Accuracy: {svmAccuracy}")

# Total waktu eksekusi
total_execution_time = time.time() - start_time
print(f"Total execution time: {total_execution_time} seconds")

# Visualisasi grafik untuk analisis beban
stages = ['Load Dataset', 'Text Cleaning', 'TF-IDF Computation', 'Sentiment Analysis', 
          'Train Naive Bayes', 'Train SVM']
times = [load_dataset_time, preprocessing_time, tfidf_time, sentiment_time, 
         train_nb_time, train_svm_time]

plt.figure(figsize=(10, 6))
plt.barh(stages, times, color='skyblue')
plt.xlabel('Time (seconds)')
plt.title('Execution Time for Each Stage')
plt.gca().invert_yaxis()
plt.show()

# Stop SparkSession
spark.stop()
