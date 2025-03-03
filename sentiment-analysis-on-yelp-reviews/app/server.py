from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the model and vectorizer
model = joblib.load('app/model.joblib')
vectorizer = joblib.load('app/vectorizer.joblib')

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Yelp Reviews model API'}

class Review(BaseModel):
    text: str

@app.post('/predict')
def predict(review: Review):
    """
    Predicts the sentiment of the review.

    Args:
        review (Review): A review text to predict.

    Returns:
        dict: A dictionary containing the predicted sentiment and confidence.
    """
    try:
        review_vector = vectorizer.transform([review.text])
        prediction = model.predict(review_vector)
        probabilities = model.predict_proba(review_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        confidence = probabilities[0][prediction[0]]
        return {"sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
