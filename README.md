
# Project Title: German Text Phrase Classifier

### Description
This project develops a machine learning model to classify German text phrases into predefined categories. It includes 
a trained model using a Linear Support Vector Classifier (LinearSVC) with TF-IDF vectorization, a REST API developed 
using FastAPI for model serving, and Docker for easy deployment. I have also added a static form where you can enter the
query and see the prediction.

## Setup and Installation
### Clone the Repository
```
git clone [repository-link]
cd [repository-name]
```
### Build and Run the Docker Container
```
docker build -t german-text-classifier .
docker run -p 8000:8000 german-text-classifier
```
### Usage
> Access the API at: http://localhost:8000   
> Use the `/predict` endpoint to classify German text phrases.   
> Send a **POST** request with a JSON body containing the text phrase.

### API Endpoints
1. **GET** `/`: Displays a simple HTML form for inputting text.
2. **POST** `/predict`: Accepts a text input and returns the classification.

### Machine Learning Pipeline    
1. **Model Selection**: _LinearSVC_ was chosen due to its efficiency with text data and its strong performance with 
high-dimensional spaces.
2. **Preprocessing**: Text data is normalized by converting to lowercase and removing non-alphabetic characters to 
standardize inputs.
3. **API Framework**: FastAPI was selected for its high performance and ease of use, especially for creating REST APIs.
4. **Dockerization**: Ensures consistent environments and simplifies deployment.          
5. **Web Form**: Added _HTML_ form ```static/form.html``` to get and display the results.

### Code Structure
`train_model.py`: Contains code for training the machine learning model and preprocessing functions.  
`app.py`: Defines the FastAPI application and endpoints.  
`test_app.py`: Includes tests for the API endpoints.  
`Dockerfile`: Instructions for Docker to build the application container.  
`requirements.txt`: Lists the Python dependencies.
`form.html`: Include the form for the better user interaction.

### Testing
```
pytest test_app.py
```

### Future Work
- Implement a more advanced NLP model like BERT, especially a variant fine-tuned for German.