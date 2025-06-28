from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import pickle
import os
from umap import UMAP
from sentence_transformers import SentenceTransformer
from utils import get_gpt_client
from bertopic.representation import OpenAI
import pandas as pd  # Import pandas
from sklearn.feature_extraction.text import CountVectorizer
import random
import numpy as np
import torch

def create_bertopic_model(data, number_of_topics=50, n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', min_topic_size=10):  
    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(52)


    print(f"Generating new model...")

    # Set up UMAP
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                min_dist=min_dist, metric=metric, random_state=42)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # Can use paraphrase-multilingual-MiniLM-L12-v2 for multilingual data

    # Custom Prompt
    prompt = """
    You will extract a short topic label from given documents and keywords. The documents are hallucination
    descriptions from a visual imagery experiment where the participant watched a Ganzflicker.
    Here are two examples of topics you created before:

    # Example 1
    Sample texts from this topic:
    - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
    - Meat, but especially beef, is the worst food in terms of emissions.
    - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

    Keywords: meat beef eat eating emissions steak food health processed chicken
    topic: Environmental impacts of eating meat

    # Example 2
    Sample texts from this topic:
    - I saw people walking around and their faces grew bigger.
    - There were a lot of people looking at me and it scared me!
    - I saw a flashing screen that had people in them and they had weird faces.
    - There were a bunch of faces floating around.

    Keywords: face faces people walking
    topic: Faces

    # Your task
    Sample texts from this topic:
    [DOCUMENTS]
    Keywords: [KEYWORDS]

    Based on the information above, extract a short topic label (1-3 words). Use Title case.
    Be concise: avoid filler words like 'imagery', 'hallucination', 'experiment', 'experiences', 'patterns', etc. in the topic label. 
    Do so in the following format:
    topic: <topic_label>
    """

    representation_model = OpenAI(get_gpt_client(), model='gpt-4o-mini', prompt=prompt) # This will improve the topic names

    # Create a BERTopic model
    # If number_of_topics is not None, we will use it to create the model
    # If number_of_topics is None, we will use the default number of topics that it creates
    if number_of_topics is not None:
        topic_model = BERTopic(representation_model=representation_model, nr_topics=number_of_topics, min_topic_size=min_topic_size, calculate_probabilities=True, umap_model=umap_model, embedding_model=embedding_model, verbose=True)

        # Fit the BERTopic model
        topics, probs = topic_model.fit_transform(data)
    else:
        topic_model = BERTopic(representation_model=representation_model, min_topic_size=min_topic_size, calculate_probabilities=True, umap_model=umap_model, embedding_model=embedding_model, verbose=True)
        # Fit the BERTopic model
        topics, probs = topic_model.fit_transform(data)

    # Replace the number of the topic with the actual topic name
    topic_df = topic_model.get_topic_info()
    topics = [topic_df[topic_df['Topic'] == i]['Name'].iloc[0] for i in topics]

    return topic_model, topics, probs, umap_model, embedding_model