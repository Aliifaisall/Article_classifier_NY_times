import gradio as gr  # Import the Gradio library
import json  # Import JSON module for working with JSON files
import os  # Import OS module for interacting with the operating system
import numpy as np  # Import NumPy for numerical computing
from PIL import Image  # Import Image module from Python Imaging Library for image processing
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF vectorizer from scikit-learn
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier from scikit-learn
# Import train_test_split and cross_val_score for splitting data and cross-validation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score  # Import accuracy_score for model evaluation
from summa import summarizer  # Import summarizer from Summa for generating abstracts
import nltk  # Import NLTK for natural language processing tasks
from nltk.corpus import stopwords  # Import stopwords from NLTK
from nltk.tokenize import word_tokenize  # Import word_tokenize from NLTK for tokenization
from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer from NLTK for lemmatization
import string  # Import string module for string operations


class ArticleClassifier:
    """
    A class to handle article classification.

    Attributes:
    _vectorizer (TfidfVectorizer): TF-IDF vectorizer for text preprocessing.
    _clf (RandomForestClassifier): Random forest classifier for category prediction.
    """
    # Function to generate abstracts

    def generate_abstract(self, text):
        return summarizer.summarize(text)

    def __init__(self):
        # Initialize TF-IDF vectorizer and random forest classifier
        self._vectorizer = None
        self._clf = None

    def train(self, X_text, y):
        # Fit the TF-IDF vectorizer with the training data
        self._vectorizer = TfidfVectorizer(max_features=500)
        X_tfidf = self._vectorizer.fit_transform(X_text)
        # Print the shape of X_tfidf
        print("Number of features:", X_tfidf.shape[1])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Train a RandomForestClassifier with cross-validation
        self._clf = RandomForestClassifier(n_estimators=1000, random_state=42)
        cv_scores = cross_val_score(self._clf, X_train, y_train, cv=5)
        print("Cross-validation scores:", cv_scores)
        print("Mean CV accuracy:", np.mean(cv_scores))

        # Fit the classifier on the training data
        self._clf.fit(X_train, y_train)
        # Evaluate the model on the test set
        y_pred = self._clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Test set accuracy:", test_accuracy)

    # Function for text preprocessing
    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in string.punctuation]
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    # Predict category for given input
    def predict_category(self, article_body="", image=None, title=""):
        if article_body:
            # Preprocess article body
            preprocessed_article = self.preprocess_text(article_body)
            # Generate abstract
            abstract = self.generate_abstract(article_body)
            # Perform image feature extraction
            if image is not None:
                image_feature = np.random.rand(1000)  # Placeholder for actual image feature extraction
            else:
                # use placeholder image feature
                image_feature = np.zeros(1000)
          # Ensure image_feature has one dimension
            image_feature = image_feature.reshape(1, -1)
          # Transform text feature
            text_feature = self._vectorizer.transform([preprocessed_article]).toarray()
          # text_feature has compatible dimensions
            if text_feature.shape[1] != 500:
                text_feature = np.zeros((1, 500))  # Placeholder for text feature
          # Combine text and image features
            input_features = np.concatenate((image_feature, text_feature), axis=1)
          # Predict category
            category = self._clf.predict(input_features.reshape(1, -1))[0]
            return category, abstract
        else:
            return "Article body not provided.", ""


# Load JSON data
with open(r'C:\git\Rihal\N24News\news\nytimes_dev.json') as f:
    data = json.load(f)

# Extract relevant images from JSON
image_folder = r'C:\git\Rihal\N24News\imgs'
image_id_to_file = {os.path.splitext(file)[0]: os.path.join(image_folder, file) for file in os.listdir(image_folder)}
image_features = []
sections = []
preprocessed_texts = []

# Construct a loop that iterates through the data
for item in data:
    section = item.get("section", "")
    headline = item.get("headline", "")
    article = item.get("article", "")

    # Preprocess text
    article_classifier = ArticleClassifier()
    preprocessed_article = article_classifier.preprocess_text(article)
    preprocessed_texts.append(preprocessed_article)
    sections.append(section)

    # Extract image features
    image_id = item.get("image_id", "")
    image_file = image_id_to_file.get(image_id, "")
    if image_file and os.path.exists(image_file):
        image = Image.open(image_file)
        image_feature = np.random.rand(1000)
        image_features.append(image_feature)
    else:
        image_features.append(np.zeros(1000))

# Convert image features to NumPy array
image_features = np.array(image_features)

# Initialize and train the ArticleClassifier
article_classifier = ArticleClassifier()
article_classifier.train(preprocessed_texts, sections)

# Gradio interface
iface = gr.Interface(
    fn=article_classifier.predict_category,
    inputs=[
        gr.Textbox(lines=10, label="Article Body"),
        gr.Image(type="pil", label="Image (Optional)"),
        gr.Textbox(label="Title (Optional)")
    ],
    outputs=[
        gr.Textbox(label="Predicted Category"),
        gr.Textbox(label="Generated Abstract")
    ],
    title="Article Categorization & Summarization",
    description="Enter an article body to predict its category and generate an abstract. Optionally, upload an image and provide a title."
)

# Launch the Gradio app
iface.launch()
