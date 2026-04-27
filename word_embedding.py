"""
Word Embedding Comparison: Word2Vec (Google) vs. FastText (Meta)
Exploring semantic relationships in 3D vector space using PCA.
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt # Library for visualization
from sklearn.decomposition import PCA
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

# Define a sample dataset
# (Feel free to use English or Turkish sentences here)
sentences = [
    "Dogs are very cute animals.",
    "Dogs are domestic animals.",
    "Cats usually love to move independently.",
    "Dogs are loyal and friendly animals."
]

# Preprocessing: Tokenize sentences into lowercased word lists
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Initialize and train Word2Vec model (Google)
# vector_size: dimensionality of the word vectors
# window: maximum distance between the current and predicted word
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

# Initialize and train FastText model (Meta/Facebook)
# FastText considers subword information (n-grams), unlike Word2Vec
fasttext_model = FastText(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

# Visualization function using PCA (Principal Component Analysis)
def plot_word_embedding(model, title):
    word_vectors = model.wv
    # Select the first 1000 words to display
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]
    
    # Apply PCA to reduce 50-dimensional vectors to 3 dimensions for visualization
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)
    
    # Initialize 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot the vectors as points in 3D space
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], color='teal')
    
    # Label each point with its corresponding word
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize=10)
    
    # Set plot titles and axis labels
    ax.set_title(f"3D Word Embedding Visualization: {title}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()

# Run the visualization for both models
plot_word_embedding(word2vec_model, "Word2Vec")
plot_word_embedding(fasttext_model, "FastText")