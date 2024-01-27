from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib  # Use joblib to load the trained model and vectorizer for making predictions
from train_model import preprocess_text  # Importing the preprocess_text from train_model.py
from fastapi.middleware.cors import CORSMiddleware  # Importing the module for CORS middleware

app = FastAPI()

# This middleware will help in handling Cross-Origin Resource Sharing CORS headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

# Loading the pre-trained model and vectorizer
try:
    model = joblib.load('trained_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_form():
    """return the html form for user input"""
    return FileResponse("static/form.html")

@app.post("/predict")
def predict(query: Query):
    if not query.text.strip():  # Checks if the text is not empty or just whitespace
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input: Text query cannot be empty."
        )
    try:
        processed_query = preprocess_text(query.text)
        prediction = make_prediction(processed_query)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {e}"
        )


def make_prediction(processed_text):
    """vectorize the text & make a prediction using the pre-trained model"""
    query_tfidf = vectorizer.transform([processed_text])
    return model.predict(query_tfidf)[0]
