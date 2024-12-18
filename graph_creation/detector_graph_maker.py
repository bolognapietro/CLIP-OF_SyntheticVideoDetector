import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

DETECTOR_TYPE = "CLIP" # Choose between "CLIP" or "AIGVDet"

# Sorting function to group videos by category
def sorting_key(filename):
    """
    Determines the sorting key for a given filename based on predefined groups.

    The function categorizes filenames into groups based on specific substrings:
    - 'CogVideoX' -> group 0
    - 'Luma' -> group 1
    - 'HunyuanVideo' -> group 2
    - 'Real' -> group 3
    - Any other substring -> group 4

    Args:
        filename (str): The name of the file to be categorized.

    Returns:
        tuple: A tuple containing the group number and the original filename.
    """
    if 'CogVideoX' in filename:
        group = 0
    elif 'Luma' in filename:
        group = 1
    elif 'HunyuanVideo' in filename:
        group = 2
    elif 'Real' in filename:
        group = 3
    else:
        group = 4  # Fallback for uncategorized files

    return (group, filename)
   
def load_data(csv_path, selected_column):
    """
    Load the prediction data from a CSV file.

    Args:
        csv_path (str): The path to the CSV file containing the prediction data.

    Returns:
        dict: A dictionary containing the prediction data.
    """
    # Load the CSV file
    data = pd.read_csv(csv_path)
    
    # Sort the data 
    data['filename'] = data['filename'].apply(lambda x: x.split('/')[-1])
    data = data.sort_values(by='filename', key=lambda col: col.map(sorting_key)).reset_index(drop=True)

    # Create a DataFrame with the filename column and the selected column
    data = pd.DataFrame({
        'filename': data['filename'],
        'prediction': data[selected_column]
    })
    return data

def compute_accuracy(data, grouped_labels):
    """
    Computes the accuracy for each category of videos.

    Args:
        data (dict): A dictionary containing the prediction results. 
                     It should have a key 'predict' with a list of prediction scores.
        grouped_labels (list): A list of labels corresponding to the categories of the videos.

    Returns:
        dict: A dictionary with the accuracy information for each category. 
              The keys are the category names ('CogVideoX', 'Luma', 'HunyuanVideo', 'Real') 
              and the values are lists where the first element is the count of correct predictions 
              and the second element is the total number of predictions for that category.
    """
    # Create a dictionary to store the accuracy for each category
    accuracy = {'CogVideoX': [0, 0], 'Luma': [0, 0], 'HunyuanVideo': [0, 0], 'Real': [0, 0]}  # Initialize accuracy info

    # Compute accuracy for each type of video
    for i, label in enumerate(grouped_labels):
        if label in accuracy:
            if (label == 'Real' and data['prediction'][i] <= 0.5) or (label != 'Real' and data['prediction'][i] > 0.5):
                accuracy[label][0] += 1
            accuracy[label][1] += 1

    return accuracy

def set_plot_params(grouped_labels, accuracy):
    """
    Set plot parameters including ranges, x-ticks, and x-positions.

    Args:
        grouped_labels (list): A list of labels grouped together.
        accuracy (dict): A dictionary where keys are labels and values are tuples 
                         containing the number of correct predictions and total predictions.

    Returns:
        tuple: A tuple containing:
            - ranges (list of tuples): Each tuple contains the start and end indices of a range.
            - x_ticks (list): A list of x-tick labels with accuracy information.
            - x_positions (list): A list of positions for the x-ticks, centered within each range.
    """
    ranges = []
    start_index = 0
    current_label = grouped_labels[0]

    # Group labels into ranges to display them in the plot
    for i, label in enumerate(grouped_labels):
        if label != current_label or i == len(grouped_labels) - 1:
            if i == len(grouped_labels) - 1:  # Include the last range
                ranges.append((start_index, i))
            else:
                ranges.append((start_index, i - 1))
                start_index = i
                current_label = label

    # Set x_ticks and x_positions for the plot based on the ranges
    x_ticks = [grouped_labels[start] for start, _ in ranges]  # One label per range
    x_positions = [(start + end) // 2 for start, end in ranges]  # Center of each range

    # Add accuracy informations to x_ticks
    for i, tick in enumerate(x_ticks):
        correct, total = accuracy[tick]
        accuracy_text = f"accuracy: {correct}/{total} ({correct/total:.2f})"
        x_ticks[i] = f"{tick}\n{accuracy_text}"

    return ranges, x_ticks, x_positions

def plot_graph(data, ranges, x_ticks, x_positions, graph_name, threshold=0.5):
    """
    Plots a graph of predictions with real and fake video ranges highlighted.

    Args:
        data (dict): Dictionary containing prediction data with key 'predict'.
        ranges (list of tuples): List of tuples indicating the start and end of ranges.
        x_ticks (list): List of labels for the x-axis ticks.
        x_positions (list): List of positions for the x-axis ticks.
        graph_name (str): Name of the graph to be used in the title and saved file.
        threshold (float, optional): Threshold value to distinguish between real and fake videos. Defaults to 0.5.
    
    Returns:
        None
    """
    # Create the plot
    plt.figure(figsize=(18, 9))

    # Add rectangles to indicate the real and fake video ranges
    plt.axhspan(0, 0.5, color='green', alpha=0.3)  # Green rectangle for values <= 0.5
    plt.axhspan(0.5, 1, color='red', alpha=0.3)    # Red rectangle for values > 0.5

    for idx, value in enumerate(data['prediction']):
        plt.plot([idx, idx], [0, value], color='blue', linewidth=1)  # Vertical line
    
    plt.scatter(range(len(data['prediction'])), data['prediction'], label='Prediction', color='blue', s=15)
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5)
    
    # Add vertical lines to separate ranges
    for _, end in ranges[:-1]:  # Skip the last range for vertical line
        plt.axvline(x=end + 0.5, color='gray', linestyle='--', linewidth=1)

    # Customize plot
    plt.title(f'Graph of {graph_name} predictions', fontsize=14)
    plt.ylabel('Prediction', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(x_positions, x_ticks, rotation=0, ha='center')  # Group labels with accuracy
    plt.grid(alpha=0.3)

    # Add legend with color indication
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=7, label='Prediction'),
        Line2D([0], [0], color='green', lw=4, label='Real Video'),
        Line2D([0], [0], color='red', lw=4, label='Fake Video')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()  # Adjust layout to prevent label clipping

    # Create the results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save and show the plot
    plt.savefig(os.path.join(os.path.dirname(__file__), f'results/{graph_name} results.png'))  # Save the figure
    plt.show()

if __name__ == '__main__':
    # Path to the CSV file (change to the desired file)
    csv_path = os.path.join(os.getcwd(), '../results/results_complete_dataset.csv')
    
    # Load the prediction data
    prediction_type = {
        0: 'prediction_clipdet_latent10k',
        1: 'prediction_clipdet_latent10k_plus',
        2: 'prediction_Corvi2023',
        3: 'prediction_fusion[mean_logit]',
        4: 'prediction_fusion[max_logit]',
        5: 'prediction_fusion[median_logit]',
        6: 'prediction_fusion[lse_logit]',
        7: 'prediction_fusion[mean_prob]',
        8: 'prediction_fusion[soft_or_prob]',
        9: 'prediction_OF'
    }

    #! Choose the prediction type to analyze (0-9) from the prediction_type dictionary
    selected_column = prediction_type[8] 
    data = load_data(csv_path, selected_column)

    # Group video labels based on their filenames
    grouped_labels = [sorting_key(video_name)[0] for video_name in data['filename']]
    label_map = {0: 'CogVideoX', 1: 'Luma', 2: 'HunyuanVideo', 3: 'Real', 4: 'Other'}
    grouped_labels = [label_map[label] for label in grouped_labels]

    # Calculate accuracy for each group
    accuracy = compute_accuracy(data, grouped_labels)

    # Set plot parameters
    ranges, x_ticks, x_positions = set_plot_params(grouped_labels, accuracy)

    # Plot the graph
    plot_graph(data, ranges, x_ticks, x_positions, selected_column)