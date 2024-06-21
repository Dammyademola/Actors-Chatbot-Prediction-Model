from tkinter import filedialog
from PIL import Image
from tensorflow import keras
import numpy as np
import tkinter as tk
import pandas
import aiml
import requests
import re
import time
import warnings
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from nltk.sem import logic

read_expr = Expression.fromstring
time.clock = time.time
warnings.filterwarnings("ignore")


# Function to get movie details from API
def get_movie_details(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    api_key = "f3ed6ba49976778cc4fc94f8ba64096f"
    params = {"api_key": api_key, "query": movie_title}

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["results"]:
                movie = data["results"][0]
                title = movie["title"]
                overview = movie["overview"]
                release_date = movie["release_date"]
                rating = movie["vote_average"]

                movie_details = (
                    "Title: {}\nOverview: {}\nRelease Date: {}\nRating: {}".format(
                        title, overview, release_date, rating
                    )
                )

                return movie_details
            else:
                return "Sorry, I couldn't find information about that movie."
        else:
            return "Sorry, there was a problem accessing the TMDb API."
    except Exception as e:
        print("Error:", e)
        return "Sorry, an error occurred while processing your request."


# Load initial KB from CSV file
kb = []
data = pandas.read_csv("Coursework/TaskB/logical-kb.csv", header=None)
[kb.append(Expression.fromstring(row)) for row in data[0]]

# Check KB integrity (no contradiction)
integrity_check_expr = logic.Expression.fromstring("all x.(all y.(x=y))")
integrity_check_result = ResolutionProver().prove(
    integrity_check_expr, kb, verbose=False
)
if not integrity_check_result:
    print("Error: Knowledge base contains contradictions.")

# Load AIML Kernel
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="Coursework/TaskB/mybot-logic.xml")

# Read Q&A knowledge base from CSV file
df = pandas.read_csv("Coursework/TaskA/Q-A_Knowledgebase.csv")
df_questions = df["Questions"]
df_answers = df["Answers"]

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(df_questions)

# Load the model
model = keras.models.load_model("Coursework/TaskC/MovieActorClassification.h5")

with open("Coursework/TaskC/actor_labels.txt", "r") as f:
    actor_labels = f.read().splitlines()


# Function to open file dialog for image selection
def upload_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open file dialog
    return file_path


# Function to predict genre from the uploaded image
def predicted_actor(image_path, actor_labels):
    img = Image.open(image_path)
    img_resized = img.resize(
        (224, 224)
    )  # Resize the image to match the expected input shape
    img_array = np.array(img_resized)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)

    # Map predicted class index to actor label
    predicted_class_index = np.argmax(prediction)
    predicted_actor_label = actor_labels[predicted_class_index]

    return predicted_actor_label


print("Welcome to this chat bot. Please feel free to ask questions from me!")

# Main conversation loop
while True:
    try:
        userInput = input("> ").lower()

        # Response Agent (AIML or Similarity-based)
        responseAgent = "aiml"
        if responseAgent == "aiml":
            answer = kern.respond(userInput)

        if answer and answer[0] == "#":
            params = answer[1:].split("$")
            cmd = int(params[0])

            if cmd == 0:
                print("Bye! Nice talking to you. You take care now.")
                break
            elif cmd == 31:
                object, subject = params[1].split(" is ")
                expr = Expression.fromstring(subject + "(" + object + ")")

                # Check for contradiction with the existing knowledge base
                contradiction_check = ResolutionProver().prove(expr, kb)
                if contradiction_check:
                    print(
                        "Error: The provided information contradicts with existing knowledge."
                    )
                else:
                    kb.append(expr)
                    print("OK, I will remember that", object, "is", subject)

            elif cmd == 32:
                object, subject = params[1].split(" is ")
                expr = Expression.fromstring(subject + "(" + object + ")")

                # Check if the statement is provable from the knowledge base
                proof = ResolutionProver().prove(expr, kb)
                if proof:
                    print("Correct.")
                elif not ResolutionProver().prove(expr.negate(), kb):
                    print("I don't know.")
                else:
                    print("Incorrect.")

            elif cmd == 99:
                print("I did not get that, please try again.")
        elif answer:
            print(answer)

        else:
            # Normalize user input
            normalized_input = re.sub(r"[^\w\s]", "", userInput.lower())

            # Check if there's an exact match
            if normalized_input in df_questions.str.lower().tolist():
                index = df_questions.str.lower().tolist().index(normalized_input)
                print(df_answers.iloc[index])
            else:
                UserAnswersVectorizerd = vectorizer.transform([normalized_input])
                QuestionsVectorizerd = vectorizer.transform(df_questions)
                Cosinesim = cosine_similarity(
                    UserAnswersVectorizerd, QuestionsVectorizerd
                )
                MostSimilar = Cosinesim.argmax()

                if Cosinesim[0][MostSimilar] > 0.5:
                    FindAnswer = df_answers.iloc[MostSimilar]
                    print(FindAnswer)
                else:
                    # If no suitable answer found, you can print a default message
                    print("Sorry, I couldn't find a relevant answer for your question.")

        if userInput.startswith("who") and userInput.endswith("actor?"):
            print("Please upload the image of the actor.")
            image_path = upload_image()
            if image_path:
                predicted_actor = predicted_actor(image_path, actor_labels)
                print("I think the Actor is: ", predicted_actor)

    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
