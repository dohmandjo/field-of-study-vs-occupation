from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
spark = SparkSession.builder.appName("ChangeCareerAPI").getOrCreate()
model = LogisticRegressionModel.load("../../career_prediction_model")

class UserInput(BaseModel):
    fieldOfStudy: str
    currentOccupation: str
    gender: str
    educationLevel: str
    industryGrowthRate: str
    familyInfluence: str
    age: int
    yearsOfExperience: int
    jobSatisfaction: int
    workLifeBalance: int
    jobOpportunities: int
    salary: int
    jobSecurity: int
    careerChangeInterest: int
    skillsGap: int
    mentorshipAvailable: int
    certifications: int
    freelancingExperience: int
    geographicMobility: int
    professionalNetworks: int
    careerChangeEvents: int
    technologyAdoption: int


app = FastAPI()

# 👇 Add this block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(user_input: UserInput):
    # Store user input in a pandas DataFrame
    df = pd.DataFrame([user_input.dict()])

    # Convert pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(df)

    # Define the categorical columns to index 
    categorical_cols = [
        "fieldOfStudy", "currentOccupation", "gender",
        "educationLevel", "industryGrowthRate", "familyInfluence"
    ]

    # Create corresponding output columns for indexers
    index_cols = [col + "_index" for col in categorical_cols]

    # Create indexers for categorical columns
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="skip") for col in categorical_cols
        ]
    
    feature_cols = index_cols + [
        "age", "yearsOfExperience", "jobSatisfaction", "workLifeBalance",
        "jobOpportunities", "salary", "jobSecurity", "careerChangeInterest", "skillsGap",
        "mentorshipAvailable", "certifications", "freelancingExperience", "geographicMobility",
        "professionalNetworks", "careerChangeEvents", "technologyAdoption"
    ]

    # Assemble all features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Create and run a pipeline (StringIndexers -> VectorAssembler)
    pipeline = Pipeline(stages=indexers + [assembler])
    transformed_df = pipeline.fit(spark_df).transform(spark_df)

    # Run prediction
    predictions = model.transform(transformed_df)

    # Extract predictions and probabilities
    row = predictions.select("prediction", "probability").toPandas().iloc[0]
    result = {
        "prediction": int(row["prediction"]),
        "probability_0": float(row["probability"][0]),
        "probability": float(row["probability"][1])
    }
    return result
    # Here you would typically load your model and make a prediction

@app.get("/favicon.ico")
async def favicon():
    return {}

