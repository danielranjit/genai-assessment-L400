import os
import yaml
import logging
import google.cloud.logging
from flask import Flask, render_template, request

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings

# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Read application variables from the config fle
BOTNAME = "FreshBot"
SUBTITLE = "Your Friendly Restaurant Safety Expert"

app = Flask(__name__)

# Initializing the Firebase client
db = firestore.Client()

# TODO: Instantiate a collection reference
collection = db.collection("food-safety")

# TODO: Instantiate an embedding model here
def text_embedding(text_to_embed) -> list:
    """Text embedding with a Large Language Model."""
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    embeddings = model.get_embeddings([text_to_embed])
    for embedding in embeddings:
        vector = embedding.values
        print(f"Length of Embedding Vector: {len(vector)}")
    return vector

# TODO: Instantiate a Generative AI model here
gen_model = GenerativeModel("gemini-1.5-pro-001")

# TODO: Implement this function to return relevant context
# from your vector database
def search_vector_database(query: str):

    query_embedding = text_embedding(query)
    # Find the 5 nearest neighbors to the query embedding
    results = collection.find_nearest(
    vector_field="embedding",
    query_vector=Vector(query_embedding),
    distance_measure=DistanceMeasure.EUCLIDEAN,
    limit=5,
    )


    # Extract the document data from the search results
    snapshots = results.get()
    context = ""
    for snapshot in snapshots:
        context += snapshot.to_dict()['content']

    # Don't delete this logging statement.
    logging.info(
        context, extra={"labels": {"service": "cymbal-service", "component": "context"}}
    )
    return context

# TODO: Implement this function to pass Gemini the context data,
# generate a response, and return the response text.
def ask_gemini(question):

    # 1. Create a prompt_template with instructions to the model
    # to use provided context info to answer the question.
    prompt_template = """
    Answer the users question using the following data.
    Only use the data provided below. Do not make anything up
    or use your own data.
    Data: {0}
    Question: {1}
    Answer:
    """

    # 2. Use your search_vector_database function to retrieve context
    # relevant to the question.
    context = search_vector_database(question)

    # 3. Format the prompt template with the question & context
    prompt = prompt_template.format(context, question)

    # 4. Pass the complete prompt template to gemini and get the text
    # of its response to return below.
    response = gen_model.generate_content(
    prompt,
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0,
        "top_p": 1
    },
    )
    return response.text

# The Home page route
@app.route("/", methods=["POST", "GET"])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == "GET":
        question = ""
        answer = "Hi, I'm FreshBot, what can I do for you?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else:
        question = request.form["input"]
        # Do not delete this logging statement.
        logging.info(
            question,
            extra={"labels": {"service": "cymbal-service", "component": "question"}},
        )
        
        # Ask Gemini to answer the question using the data
        # from the database
        answer = ask_gemini(question)

    # Do not delete this logging statement.
    logging.info(
        answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}}
    )
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
    }

    return render_template("index.html", config=config)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
