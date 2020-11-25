
# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/usr/lib/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path


from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint


def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    return LabeledPoint(label, feats)

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("hdfs:///data")
parsedData = data.map(mapper)

# Train model
model = LogisticRegressionWithSGD.train(parsedData)

# Predict the first elem will be actual data and the second 
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: ((point.label), 
        model.predict(point.features)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda p: p[0] != p[1]).count() / float(parsedData.count())

# Print some stuff
print("LogisticRegressionWithSGD result:")
print("Training Error = " + str(trainErr))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

w = np.zeros(4)
lr = 1e-3

for i in range(100):
    w += lr * parsedData.map(lambda p: sigmoid(-p.label * np.dot(p.features, w)) * p.label * p.features).reduce(lambda a, b: a + b)

labelsAndPreds = parsedData.map(lambda point: ((point.label), 
        1 if sigmoid(np.dot(point.features, w)) > 0.5 else 0))

trainErr = labelsAndPreds.filter(lambda p: p[0] != p[1]).count() / float(parsedData.count())

print("my gradient descent result:")
print("Training Error = " + str(trainErr))


