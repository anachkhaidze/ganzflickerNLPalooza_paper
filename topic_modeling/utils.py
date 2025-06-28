import pandas as pd
from scipy.spatial import distance
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import language_tool_python
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from bertopic import BERTopic
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import numpy as np

# Get hallucination data from data/hallucinations.csv and return the pandas df
def get_hallucination_data():
    df = pd.read_csv('data/hallucinations.csv')
    return df


# Get hallucination descriptions from data/hallucinations.csv
def get_descriptions():
    score_question = "How would you describe your VISUAL imagery vividness on a scale from 0-10?"
    describe_question = "Please describe as much as you can remember about what you saw in the Ganzflicker:"
    df = get_hallucination_data()[[score_question, describe_question]].dropna()
    return df[describe_question].tolist()


# Get hallucination scores from data/hallucinations.csv
def get_scores():
    score_question = "How would you describe your VISUAL imagery vividness on a scale from 0-10?"
    describe_question = "Please describe as much as you can remember about what you saw in the Ganzflicker:"
    df = get_hallucination_data()[[score_question, describe_question]].dropna()
    return df[score_question].tolist()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text


def spellcheck_list(text_list):
    tool = language_tool_python.LanguageTool('en-US')
    corrected_list = []
    for text in text_list:
        matches = tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        corrected_list.append(corrected)
    return corrected_list


def get_gpt_client():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAIAPI_KEY"))
    return client


def get_gpt_response(prompt, temperature=None, model="gpt-4o-mini"):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAIAPI_KEY"))
    messages = [
        {"role": "user", "content": prompt},
    ]

    if model == "o3-mini":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
    response_message = response.choices[0].message.content
    return response_message


def calculate_coherence_score(topic_model, docs, top_n_words=10, coherence='c_v'):
    """
    Calculate topic coherence for a BERTopic model.
    Parameters:
    - topic_model: a fitted BERTopic model
    - docs: original list of documents (list of strings)
    - top_n_words: number of top words to consider per topic
    - coherence: coherence metric to use ('u_mass', 'c_v', 'c_uci', 'c_npmi')
    Returns:
    - Coherence score (float)
    """
    # Preprocess the documents using BERTopic's internal method
    cleaned_docs = topic_model._preprocess_text(docs)
    
    # Get the tokenizer from the vectorizer
    vectorizer = topic_model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()
    
    # Tokenize the documents
    tokenized_docs = [tokenizer(doc) for doc in cleaned_docs]
    
    # Create a Gensim dictionary and corpus
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    
    print("Extracting topic words from c-TF-IDF matrix...")
    
    # Use c-TF-IDF matrix to get actual vocabulary words (not labels)
    try:
        # Get topic-word matrix
        topic_word_matrix = topic_model.c_tf_idf_.toarray()
        feature_names = topic_model.vectorizer_model.get_feature_names_out()
        
        print(f"Topic-word matrix shape: {topic_word_matrix.shape}")
        print(f"Number of vocabulary features: {len(feature_names)}")
        
        topic_words = []
        
        # Get topics (skip outlier topic -1 if it exists)
        topics = topic_model.get_topics()
        valid_topic_ids = [tid for tid in topics.keys() if tid != -1]
        
        print(f"Valid topic IDs: {valid_topic_ids}")
        
        for topic_id in valid_topic_ids:
            if topic_id < topic_word_matrix.shape[0]:
                # Get topic scores
                topic_scores = topic_word_matrix[topic_id]
                
                # Get top word indices for this topic
                top_word_indices = topic_scores.argsort()[-top_n_words:][::-1]
                
                # Get actual words with positive scores
                words = []
                for idx in top_word_indices:
                    if topic_scores[idx] > 0:
                        word = feature_names[idx]
                        words.append(word)
                
                if words:  # Only add if we have valid words
                    topic_words.append(words)
                    print(f"Topic {topic_id}: {words[:5]}...")  # Show first 5 words
        
        print(f"Successfully extracted {len(topic_words)} topics with vocabulary words")
        
        # Calculate coherence
        if topic_words and len(topic_words) > 1:  # Need at least 2 topics
            print(f"Calculating coherence with {len(topic_words)} topics...")
            
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence=coherence
            )
            coherence_score = coherence_model.get_coherence()
            return coherence_score
        else:
            print(f"Insufficient topics for coherence calculation. Found: {len(topic_words)}")
            return None
            
    except Exception as e:
        print(f"Error extracting topics from c-TF-IDF matrix: {e}")
        
        # Fallback: Try to tokenize the topic labels
        print("\nFallback: Tokenizing topic labels...")
        try:
            topics = topic_model.get_topics()
            topic_words = []
            
            for topic_id in topics.keys():
                if topic_id != -1:  # Skip outlier topic
                    topic_repr = topic_model.get_topic(topic_id)
                    if topic_repr:
                        # Extract labels and tokenize them
                        topic_tokens = []
                        for item in topic_repr[:top_n_words]:
                            if isinstance(item, list) and len(item) > 0:
                                label = item[0]  # Get the label
                                # Tokenize the label into individual words
                                tokens = tokenizer(label.lower())
                                topic_tokens.extend(tokens)
                        
                        # Remove duplicates while preserving order
                        unique_tokens = []
                        seen = set()
                        for token in topic_tokens:
                            if token not in seen and len(token) > 1:  # Skip single characters
                                unique_tokens.append(token)
                                seen.add(token)
                        
                        if unique_tokens:
                            topic_words.append(unique_tokens[:top_n_words])
                            print(f"Topic {topic_id} tokenized: {unique_tokens[:5]}...")
            
            if topic_words and len(topic_words) > 1:
                print(f"Calculating coherence with {len(topic_words)} tokenized topics...")
                coherence_model = CoherenceModel(
                    topics=topic_words,
                    texts=tokenized_docs,
                    dictionary=dictionary,
                    coherence=coherence
                )
                coherence_score = coherence_model.get_coherence()
                return coherence_score
            else:
                print("Fallback method also failed")
                return None
                
        except Exception as e2:
            print(f"Fallback method failed: {e2}")
            return None


def get_topic_info_summary(topic_model):
    """
    Helper function to understand your BERTopic model structure
    """
    print("=== BERTopic Model Summary ===")
    
    # Basic info
    topics = topic_model.get_topics()
    print(f"Number of topics: {len(topics)}")
    print(f"Topic IDs: {list(topics.keys())}")
    
    # Check for topic info
    try:
        topic_info = topic_model.get_topic_info()
        print(f"Topic info shape: {topic_info.shape}")
        print("Topic info columns:", topic_info.columns.tolist())
        print("\nFirst few topics:")
        print(topic_info.head())
    except:
        print("No topic_info available")
    
    # Check vectorizer
    if hasattr(topic_model, 'vectorizer_model'):
        try:
            vocab_size = len(topic_model.vectorizer_model.get_feature_names_out())
            print(f"Vocabulary size: {vocab_size}")
        except:
            print("Could not get vocabulary size")
    
    # Check c-TF-IDF
    if hasattr(topic_model, 'c_tf_idf_'):
        print(f"c-TF-IDF matrix shape: {topic_model.c_tf_idf_.shape}")
    
    print("=" * 30)


def get_topic_map(bertopic_model, include_outlier=True):
    topic_map = {}
    
    for row_tuple in bertopic_model.get_topic_info().iterrows():
        series_data = row_tuple[1]  # Get the Series object from the tuple
        topic_id = series_data['Topic']
        
        # Skip outlier topic if include_outlier is False
        if not include_outlier and topic_id == -1:
            continue
            
        topic_name = series_data['Name'][series_data['Name'].index('_')+1:]
        topic_map[topic_id] = topic_name
    
    return topic_map