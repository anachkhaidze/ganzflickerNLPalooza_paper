import pickle
import random
import numpy as np
import torch
from bertopic import BERTopic
from bertopic.representation import OpenAI
from umap import UMAP
from sentence_transformers import SentenceTransformer
from utils import get_gpt_client 

def create_bertopic_model_and_save_keywords(data, number_of_topics=50, n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', min_topic_size=10):
    """
    Creates and fits a BERTopic model, applies custom OpenAI labels,
    and extracts the original keyword representations for coherence calculation.
    """
    # Set seeds for reproducibility
    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(52)

    print("Generating new model...")

    # --- Setup UMAP and Embedding Models ---
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                min_dist=min_dist, metric=metric, random_state=42)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Step 1: Fit a base BERTopic model first ---
    # We do this without the representation model to get the underlying c-TF-IDF keywords.
    print("Fitting BERTopic model to find base topics...")
    topic_model = BERTopic(
        nr_topics=number_of_topics, 
        min_topic_size=min_topic_size, 
        calculate_probabilities=True, 
        umap_model=umap_model, 
        embedding_model=embedding_model, 
        verbose=True
    )
    topics, probs = topic_model.fit_transform(data)

    # --- Step 2: Extract original keywords before they are replaced ---
    # This is the crucial data needed for calculating the coherence score later.
    print("Extracting original c-TF-IDF keywords...")
    original_keywords = topic_model.topic_representations_.copy()

    # --- Step 3: Apply the custom OpenAI labels ---
    # Now we update the topic representations with nice, human-readable names.
    print("Updating topic representations with OpenAI labels...")
    
    # This new, more direct prompt fixes the issue where the AI would give
    # conversational replies instead of topic labels.
    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]

    The topic is represented by these keywords: [KEYWORDS]

    Based on the documents and keywords, your task is to create a concise, 1-3 word topic label in Title Case.

    - Do not ask for more information.
    - Do not add any commentary or introductory text.
    - Provide only the topic label.

    The topic label is:
    """
    
    # Initialize the OpenAI representation model with the corrected prompt and chat=True
    representation_model = OpenAI(get_gpt_client(), model='gpt-4o-mini', prompt=prompt, chat=True)
    
    # Update the topics with the new labels. This doesn't re-train the model.
    topic_model.update_topics(data, topics, representation_model=representation_model)
    
    # --- Step 4: Return all necessary objects ---
    # We return the embedding_model so it can be passed to the .save() function.
    return topic_model, topics, probs, original_keywords, umap_model, embedding_model