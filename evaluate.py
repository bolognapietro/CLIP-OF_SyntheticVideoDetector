import os
import pandas as pd

parent_path = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(parent_path, "results")

os.makedirs(RESULTS_PATH, exist_ok=True)

def split_dataset(input_file):
    """
    Splits the dataset into separate files based on filename patterns.

    Parameters:
    -----------
    input_file : str
        Path to the input CSV file.

    Returns:
    --------
    dict
        A dictionary mapping descriptive keys to the paths of the generated CSV files.
    """
    
    # Load the dataset
    df = pd.read_csv(input_file)

    # Filter datasets based on filename patterns
    subsets = {
        "CogVideoX-5B": df[df['filename'].str.contains('_CogVideoX', case=False, na=False)],
        "Luma": df[df['filename'].str.contains('_Luma', case=False, na=False)],
        "Hunyuan": df[df['filename'].str.contains('_HunyuanVideo', case=False, na=False)],
        "Real": df[df['filename'].str.contains('_Real', case=False, na=False)],
        "Complete Dataset": df  
    }

    # Save subsets to CSV files
    csv_paths = {}
    for name, subset in subsets.items():
        csv_path = os.path.join(RESULTS_PATH, f'results_{name.lower().replace(" ", "_")}.csv')
        subset.to_csv(csv_path, index=False)
        csv_paths[name] = csv_path

    return csv_paths

def compute_averages(input_file, generator_name):
    """
    Computes column-wise averages and saves them to a results_avg.csv file.

    Parameters:
    -----------
    input_file : str
        Path to the input CSV file.
    generator_name : str
        Name of the generator to include in the 'generator' column.
    """
    
    df = pd.read_csv(input_file)

    # Compute averages (exclude 'filename' column)
    columns_to_average = df.columns[1:]
    averages = df[columns_to_average].mean().tolist()

    # Create a DataFrame for averages
    results_avg = pd.DataFrame([averages], columns=columns_to_average)
    results_avg.insert(0, "generator", generator_name)

    # Append the results to a master averages file
    output_file = os.path.join(RESULTS_PATH, 'results_avg.csv')
    if not os.path.exists(output_file):
        results_avg.to_csv(output_file, index=False)
    else:
        results_avg.to_csv(output_file, mode='a', header=False, index=False)

    print(f"Averages for {generator_name} saved to 'results_avg.csv'")

def main():
    input_file = os.path.join(RESULTS_PATH, 'results_complete_dataset.csv')  

    # Split the dataset into subsets
    list_csvs = split_dataset(input_file)

    # Compute averages for each subset
    for generator_name, csv_path in list_csvs.items():
        compute_averages(csv_path, generator_name)

if __name__ == "__main__":
    main()
