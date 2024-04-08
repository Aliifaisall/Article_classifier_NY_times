# Article_classifier_NY_times
an application that uses a machine learning algorithm to identify the catgeroy of diffrent news article 

Step 1:
clone the rep into your local machine and navigate to the folder through the backened.

Step 2:
**you are likly required to run the following to launch the application properly **

gradio: pip install gradio
json: This is a built-in Python library and does not require installation separately.
os: This is a built-in Python library and does not require installation separately.
numpy: pip install numpy
PIL: This is part of the Pillow library, so you can install it using pip install Pillow
scikit-learn: pip install scikit-learn
summa: pip install summa
nltk: pip install nltk (You might also need to download additional NLTK data using nltk.download() after installation)
string: This is a built-in Python library and does not require installation separately.

Step 3: 
in your terminal run -> Python gardio_app.py 


Limitation: 
I tried achiving the Mastery Level of this challenege 
Level 4: The Mastery (20 Bonus Points)
Implement a real-time UI web app for inference where it allows the user to upload an article body, 
image, and a title, and then return its category, a caption, and an abstract (you could use tools such as Streamlit or Gradio).

I was facing an Error that consumed most of my time:
ValueError : X has 1500 features, but RandomForestClassifier is expecting 500 features as input.
it always seems to be 1000 features off of RandomForestClassifier's expected features. 

this is was a challenge for me that I enjoyed and will still be working on improving this application in the future on my own. 

