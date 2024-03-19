from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory

import gensim.downloader as api
import numpy as np
from Levenshtein import distance
from sklearn.cluster import KMeans

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Get the list of words in the game from the file.
def get_words(filename: str) -> list:
    with open(filename, "r") as file:
        text = file.read()

    words = text.split(',')

    return [word.strip().lower() for word in words]


# Obtain the word embeddings for the list of words.
def embed_words(words: list, model) -> np.ndarray[float]:
    return [model[word] for word in words]


# Obtain clusters of the words, based on semantic similarity. This can be helpful in grouping words together based on meaning, although meaning could be misleading.
def get_clusters(data: np.ndarray[float], n_clusters: int) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)

    return kmeans

words = get_words("input.txt")
w2v_model = api.load("word2vec-google-news-300")
if __name__ == "__main__":

    """
    feature to add: Finding other words (limited by a pre-defined dictionary) that are linked to several of the words in the list.
    """ 

    # Define the system message to be displayed at the start of the conversation.
    system_message = f"""
    You are a chatbot helper for the word game "connections," where, provided a list of words, you must find the connection between them. 
    The game is very difficult, and often the connections are significantly more complicated than appears at first glance. In fact, you are quite susceptible to erroneous connections, and it is important to be careful in your responses.
    
    Typically, there are 16 words, and four groups of four words that are connected in some way. The goal of the game is to find the connection between the words and group them accordingly.

    Your task is to help the user. Do not solve the answer out right, but aid them based on their requests and responses. Help the user out with AT MOST one group of words at a time.

    You have several tools at your disposal, which you will have to take advantage of to properly help the user and figure out the connections yourself:
    - You can leverage some functions for:
        - Finding the levenstein edit distance between words.
        - Obtaining clusters of words based on their semantic meaning.
    - You can leverage some external resources for:
        - You can ask the user for more information, such as what they think the connection is, or what they've already tried.
    
    The words are:
    {words}
    """

    """
    Tool to find the Levenshtein edit distance between two words.
    """
    class LevenshteinDistanceTool(BaseTool):
        name = "Levenshtein distance"
        description = "Return the Levenshtein distance between two words."

        def _run(self, word1: str, word2: str) -> int:
            return distance(word1, word2)

    """
    Tool to find the most similar words based on semantic similarity.
    """
    class MostRelevantWordsTool(BaseTool):
        name = "Most relevant words"
        description = "Return the most relevant words to the provided input word, sorted in descending order based on semantic similarity."

        words = words
        w2v_model = w2v_model

        def _run(self, word: str) -> list[str]:

            global words, w2v_model

            def cosine_similarity(v1, v2):
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            similarities = [cosine_similarity(w2v_model[word], w2v_model[other_word]) for other_word in words]

            return [words[i] for i in np.argsort(similarities)[::-1]]

    """
    Tool to cluster the words based on semantic similarity. LLM gets to pick the number of clusters.
    """
    class SemanticClusterTool(BaseTool):
        name = "Cluster based on semantic similarity"
        description = "This can be helpful in grouping words together based on meaning, although meaning could be misleading. Pass the number of clusters as an argument - assume the function already has the words."

        def _run(self, n_clusters: int) -> str:

            global words, w2v_model

            n_clusters = int(n_clusters)

            data = embed_words(words, w2v_model)
            kmeans = get_clusters(data, n_clusters)

            labels = kmeans.labels_
            clusters = {}
            for cluster in range(n_clusters):
                cluster_words = [words[i] for i in range(len(words)) if labels[i] == cluster]
                clusters[cluster] = cluster_words

            # format clusters to be a string
            s = ""
            for cluster, words in clusters.items():
                s += f"Cluster {cluster+1}: {' '.join(words)}\n"
            
            return s

    tools = [MostRelevantWordsTool(), SemanticClusterTool()]

    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    llm = ChatOpenAI(api_key=API_KEY, model="gpt-4", temperature=0.75, verbose=True)

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        agent_kwargs={
            "system_message": system_message
        },
        verbose=True,
        memory=memory
    )

    result = agent.invoke({"input": "Provide the user a warm greeting."})
    print("Helper: ", result['output'])

    while True:
        human_input = input("User: ")

        if human_input == "/exit":
            exit()

        result = agent.invoke({"input": human_input})
        print("Helper: ", result['output'])