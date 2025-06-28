import pandas as pd
from scipy.spatial import distance
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import shutil

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


# Get hallucination descriptions from data/hallucinations.csv
def get_scores():
    score_question = "How would you describe your VISUAL imagery vividness on a scale from 0-10?"
    describe_question = "Please describe as much as you can remember about what you saw in the Ganzflicker:"
    df = get_hallucination_data()[[score_question, describe_question]].dropna()
    return df[score_question].tolist()


# Cosine distance between two vectors
def cosine_distance(vec1, vec2):
    return distance.cosine(vec1, vec2)


# Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def pearson_distance(vec1, vec2):
    """Returns 1 - Pearson correlation coefficient as a distance metric."""
    if np.all(vec1 == vec1[0]) or np.all(vec2 == vec2[0]):
        # Avoid division by zero if constant vector
        return 1.0
    corr = np.corrcoef(vec1, vec2)[0, 1]
    return 1 - corr


def dict_to_ordered_list(d):
    # Get the maximum key value to determine the list size
    max_key = max(d.keys()) + 1
    
    # Create a list with None placeholders
    result = [None] * max_key
    
    # Fill in the values at their respective positions
    for key, value in d.items():
        result[key] = value
    
    return result


def plot_rdm(vector_list, model_name, labels, show=True, distance_fn=None):
    """
    Creates and displays a Representational Dissimilarity Matrix (RDM) for model outputs.

    vector_list is a list of lists. It'll compute the similarity of each list with every other list.

    model_name is the name of the model used.
    
    distance_fn is the function used to compute distances between vectors. Defaults to cosine_distance.
    """
    if distance_fn is None:
        distance_fn = cosine_distance

    # Project model outputs to a common dimensional space
    print(f"{model_name} size: {vector_list.shape[0]}x{vector_list.shape[1]}")

    # Compute distances between model outputs
    n = vector_list.shape[0]
    rdm = np.zeros((n, n))

    # Adjust the RDM computation to align the identity-like structure from top-left to bottom-right
    for i in range(n):
        for j in range(n):
            rdm[i, j] = distance_fn(vector_list[i], vector_list[j])

    # Plot the RDM
    plt.figure(figsize=(8, 6))
    plt.imshow(rdm, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Distance')
    plt.title(f'{model_name} Representational Dissimilarity Matrix (RDM)')
    plt.xlabel('Imagery Score')
    plt.ylabel('Imagery Score')

    # Update axis labels to match the flipped structure
    if n > 12:
        unique_ordered_labels = [str(i) for i in sorted([int(i) for i in set(labels)])]
        label_spots = []
        # Find where the numbers change - labels is already sorted
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                label_spots.append(i)
        label_spots.append(len(labels)-1)
    
    else:
        unique_ordered_labels = [f"{i}" for i in np.arange(11)]
        label_spots = np.arange(11)

    plt.xticks(ticks=label_spots, labels=unique_ordered_labels)
    plt.yticks(ticks=label_spots, labels=unique_ordered_labels)     
    # Move x-axis ticks to the top of the chart
    # plt.gca().xaxis.set_ticks_position('top')
    # plt.gca().xaxis.set_label_position('top')

    # Save the RDM plot to results/plots/single folder
    results_dir = 'results/plots/single'
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f"{model_name.replace(' ', '_')}_rdm_plot.png")
    plt.savefig(plot_path)
    print(f'RDM plot saved to {plot_path}')

    if show:
        plt.show()


def display_images_grid(distance_method=None):
    """
    Display PNG images from results/plots/single/ (single-model RDMs) and results/plots/combined/ (multi-model grids).
    The plot titles will include the distance method if provided.
    """
    # Define the directory paths
    base_dir = "results/plots/"
    combined_dir = os.path.join(base_dir, "combined")
    single_dir = os.path.join(base_dir, "single")
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(single_dir, exist_ok=True)

    # Determine output filenames for combined plots
    if distance_method:
        random_output_filename = f"random_plots_{distance_method}.png"
        non_random_output_filename = f"non_random_plots_{distance_method}.png"
    else:
        random_output_filename = "random_plots.png"
        non_random_output_filename = "non_random_plots.png"
    random_output_path = os.path.join(combined_dir, random_output_filename)
    non_random_output_path = os.path.join(combined_dir, non_random_output_filename)

    # Get all PNG files in the single directory (single-model RDMs)
    single_png_files = [f for f in os.listdir(single_dir) if f.endswith('.png')]
    if not single_png_files:
        print("No single-model PNG images found in results/plots/single/ directory.")
    else:
        print(f"Found {len(single_png_files)} single-model RDM PNGs in results/plots/single/.")

    # Split single PNGs into random and non-random
    random_files = [f for f in single_png_files if 'random' in f.lower()]
    non_random_files = [f for f in single_png_files if 'random' not in f.lower()]

    # Helper to create and display a grid of images
    def create_and_show_grid(image_files, output_path, title, figsize=(12, 8)):
        if not image_files:
            return
        num_images = len(image_files)
        num_cols = 3
        num_rows = math.ceil(num_images / num_cols)
        fig = plt.figure(figsize=(figsize[0], 3 * num_rows))
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(single_dir, img_file)
            img = mpimg.imread(img_path)
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for the suptitle
        if distance_method:
            fig.suptitle(f'{title} (Distance: {distance_method})', fontsize=16, y=0.98)
        else:
            fig.suptitle(title, fontsize=16, y=0.98)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Combined {title.lower()} saved to {output_path}")
        plt.close(fig)

    # Create and display combined random grid
    create_and_show_grid(random_files, random_output_path, 'Randomized Models')
    # Create and display combined non-random grid
    create_and_show_grid(non_random_files, non_random_output_path, 'Non-Randomized Models')

    # Get all PNG files in the combined directory (multi-model grids)
    combined_png_files = [f for f in os.listdir(combined_dir) if f.endswith('.png')]
    if not combined_png_files:
        print("No combined PNG images found in results/plots/combined/ directory.")
    else:
        print(f"Found {len(combined_png_files)} combined grid PNGs in results/plots/combined/.")
        # Display the combined PNGs (multi-model grids)
        for img_file in combined_png_files:
            img_path = os.path.join(combined_dir, img_file)
            img = mpimg.imread(img_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
