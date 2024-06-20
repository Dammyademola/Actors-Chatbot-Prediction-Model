import pandas as pd
import aiml
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_movie_details(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    api_key = "f3ed6ba49976778cc4fc94f8ba64096f"
    params = {
        "api_key": api_key,
        "query": movie_title
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                movie = data['results'][0]
                title = movie['title']
                overview = movie['overview']
                release_date = movie['release_date']
                rating = movie['vote_average']
                
                movie_details = "Title: {}\nOverview: {}\nRelease Date: {}\nRating: {}".format(title, overview, release_date, rating)
                
                return movie_details
            else:
                return "Sorry, I couldn't find information about that movie."
        else:
            return "Sorry, there was a problem accessing the TMDb API."
    except Exception as e:
        print("Error:", e)
        return "Sorry, an error occurred while processing your request."
    
kern = aiml.Kernel()
kern.setTextEncoding(None)

df = pd.read_csv("Coursework/TaskA/Q-A_Knowledgebase.csv")
df_questions = df["Questions"]
df_answers = df["Answers"]

vectorizer = TfidfVectorizer()
vectorizer.fit(df_questions)

print("Welcome to this chat bot. Please feel free to ask questions from me!")


while True:
    try:
        userInput = input("> ")

        UserAnswersVectorizerd = vectorizer.transform([userInput])
        QuestionsVectorizerd = vectorizer.transform(df_questions)
        Cosinesim = cosine_similarity(UserAnswersVectorizerd, QuestionsVectorizerd)
        MostSimilar = Cosinesim.argmax()
        FindAnswer = df_answers.iloc[MostSimilar]
        print(FindAnswer)

    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    
    except Exception as e:
        print("Sorry, I encountered an error. Please try again.")
