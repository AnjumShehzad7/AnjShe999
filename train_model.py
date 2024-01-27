import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re


def load_stopword(file_path):
    """Load stop words from a file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = set(line.strip() for line in file)
    return stop_words


def preprocess_text(text_input, stopword_file_path=r'stopwords-de.txt'):
    """Preprocess input by filling missing values,
    converting to lowercase, and removing non-alphabetic characters"""
    stop_words = load_stopword(stopword_file_path)

    if isinstance(text_input, pd.Series):
        text_input = text_input.fillna('')  # Fill NaNs with empty string
        text_input = text_input.apply(lambda x: preprocess_single_text(x, stop_words))
    else:
        text_input = preprocess_single_text(text_input, stop_words)

    return text_input


def preprocess_single_text(text, stop_words):
    """Preprocess a single text"""
    text = text.lower()
    text = re.sub(r'\d', '', text)  # Remove numbers
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


def load_dataset(filepath):
    """Loads the dataset from a CSV file and preprocesses the text and labels"""
    df = pd.read_csv(filepath)
    df['text'] = preprocess_text(df['text'])

    df = df.dropna(subset=['label'])
    return df


def train_model(X_train, X_test, y_train, y_test):
    """Trains the LinearSVC model and evaluates its performance"""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, 'trained_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

    y_pred = model.predict(X_test_tfidf)
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    # final classification report
    # print(classification_report(y_test, y_pred))

def main():
    """Main function to execute the training, loading dataset & evaluation process"""
    df = load_dataset(r'sample_data.csv')

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)
    train_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
