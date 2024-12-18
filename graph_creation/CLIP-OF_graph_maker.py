import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.special import logsumexp, logit, log_expit
import os

# Define custom activation functions
softplusinv = lambda x: np.log(np.expm1(x))  # log(exp(x)-1)
softminusinv = lambda x: x - np.log(-np.expm1(x)) # info: https://jiafulow.github.io/blog/2019/07/11/softplus-and-softminus/

# weights = np.array([0.4, 0.6])
fusion_functions = {
    'mean_logit'   : lambda x, axis: np.mean(x, axis),
    'max_logit'    : lambda x, axis: np.max(x, axis),
    'median_logit' : lambda x, axis: np.median(x, axis),
    'lse_logit'    : lambda x, axis: logsumexp(x, axis),
    'mean_prob'    : lambda x, axis: softminusinv(logsumexp(log_expit(x), axis) - np.log(x.shape[axis])),
    'soft_or_prob' : lambda x, axis: -softminusinv(np.sum(log_expit(-x), axis)),
    # 'w_mean'       : lambda x, axis: np.dot(x, weights)
}

def apply_fusion(x, typ, axis):
    return fusion_functions[typ](x, axis)

def apply_mean(val1, val2):
    return (val1 + val2) / 2

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

def load_data(csv_path, prediction_type):
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

    return data['filename'], data[prediction_type]
 

def compute_accuracy(data, grouped_labels, threshold):
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
    accuracy = {}
    for label in set(grouped_labels):
        accuracy[label] = [0, 0]

    # Compute accuracy for each type of video
    for i, label in enumerate(grouped_labels):
        if label in accuracy:
            if (label == 'Real' and data['prediction'][i] <= threshold) or (label != 'Real' and data['prediction'][i] > threshold):
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

    # Extract the prediction data
    data = data['prediction']

    # Create the plot
    plt.figure(figsize=(18, 9))

    # Add rectangles to indicate the real and fake video ranges
    plt.axhspan(0, threshold, color='green', alpha=0.3)  # Green rectangle for values <= 0.5
    plt.axhspan(threshold, 1, color='red', alpha=0.3)    # Red rectangle for values > 0.5

    for idx, value in enumerate(data):
        plt.plot([idx, idx], [0, value], color='blue', linewidth=1)  # Vertical line
    
    plt.scatter(range(len(data)), data, label='CLIP-OF Prediction', color='blue', s=15)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
    
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
    csv_path = os.path.join(os.getcwd(), '../results/results_complete_dataset.csv')
    
    # Load the prediction data
    filenames_of, data_of = load_data(csv_path, prediction_type='prediction_OF')
    filenames_clip, data_clip = load_data(csv_path, prediction_type='prediction_fusion[soft_or_prob]')

    # Combine the data from both models
    fusion_methods = ['mean_logit', 'max_logit', 'median_logit', 'lse_logit', 'mean_prob', 'soft_or_prob']
    selected_fusion = 0

    rows = []
    for i in range(len(data_of)):
        if filenames_of[i] != filenames_clip[i]:
            print("Error: filenames do not match --> ", filenames_of[i], filenames_clip[i])
        else:
            result = apply_fusion(np.array([data_of[i], data_clip[i]]), fusion_methods[selected_fusion], -1)
            result = max(0, min(1, result))  # Ensure result is clamped between 0 and 1

            # Append to the temporary list
            rows.append({'filename': filenames_of[i], 'prediction': result})

    data_final = pd.DataFrame(rows)    

    # Group video labels based on their filenames
    grouped_labels = [sorting_key(video_name)[0] for video_name in data_final['filename']]
    label_map = {0: 'CogVideoX', 1: 'Luma', 2: 'HunyuanVideo', 3: 'Real', 4: 'Other'}
    grouped_labels = [label_map[label] for label in grouped_labels]

    # Calculate accuracy for each group
    threshold = 0.45
    accuracy = compute_accuracy(data_final, grouped_labels, threshold)

    # Set plot parameters
    ranges, x_ticks, x_positions = set_plot_params(grouped_labels, accuracy)

    # Plot the graph
    plot_graph(data_final, ranges, x_ticks, x_positions, 'CLIP-OF', threshold)