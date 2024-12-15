import os
import cv2
import tqdm
import yaml
import torch
import pandas as pd
import numpy as np
from PIL import Image
from optical_flow.optical_flow import *
from utils.fusion import apply_fusion
from utils.processing import make_normalize

from networks import create_architecture, load_weights
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode

parent_path = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(parent_path, "results")
JUST_SOFT_OR_PROB = False                   

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

weights_dir = os.path.join(parent_path, "weights")
temp_dir = os.path.join(parent_path, "temp_frames")
models = ['clipdet_latent10k', 'clipdet_latent10k_plus', 'Corvi2023']
fusion_methods = ['mean_logit', 'max_logit', 'median_logit', 'lse_logit', 'mean_prob', 'soft_or_prob']


def get_config(model_name, weights_dir='./weights'):
    """
    Loads model configuration and computes the path to its weights.

    Parameters:
    -----------
    model_name : str
        The name of the model whose configuration is to be loaded.
    weights_dir : str, optional
        Directory containing the model's configuration and weights (default is './weights').

    Returns:
    --------
    tuple
        A tuple containing model details: (model_name, model_path, architecture, norm_type, patch_size).
    """

    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)

    model_path = os.path.join(weights_dir, model_name, data['weights_file'])

    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

def extract_frames_from_video(video_path, output_dir):
    """
    Extract frames from an MP4 video and save them as images.

    Parameters:
    -----------
    video_path : str
        Path to the input video file.
    output_dir : str
        Directory where the extracted frames will be saved.

    Returns:
    --------
    list
        A list of file paths to the saved frame images.
    """

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_paths.append(frame_filename)
        frame_count += 1
    
    cap.release()
    return frame_paths

def generate_csv_from_frames(frame_paths, csv_path):
    """
    Create a CSV file with the list of frame image paths.
    """
    
    data = {'filename': frame_paths}
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

def running_tests(input_csv, weights_dir, models_list, device, batch_size=1):
    """
    Run inference tests using a list of models on a dataset of images.

    Parameters:
    -----------
    input_csv : str
        Path to a CSV file containing a 'filename' column with paths to the images.
    weights_dir : str
        Directory where model weights and configurations are stored.
    models_list : list
        List of model names to load and evaluate.
    device : torch.device
        The device to run the models on (e.g., 'cpu' or 'cuda').
    batch_size : int, optional
        Number of images to process in a batch (default is 1).

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the input data with additional columns 
        for each model's inference results.
    """
    
    table = pd.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    models_dict = dict()
    transform_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = 'none_%s' % norm_type
        elif patch_size == 'Clip224':
            print('input resize:', 'Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = 'Clip224_%s' % norm_type
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = 'res%d_%s' % (patch_size[0], norm_type)
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = 'crop%d_%s' % (patch_size, norm_type)
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    ### TEST
    with torch.no_grad():
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        print(do_models)
        print(do_transforms)
        print(flush=True)
        
        print("Running the Tests")
        batch_img = {k: list() for k in transform_dict}
        batch_id = list()
        last_index = table.index[-1]
        for index in tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])
            for k in transform_dict:
                batch_img[k].append(transform_dict[k](Image.open(filename).convert('RGB')))
            batch_id.append(index)

            if (len(batch_id) >= batch_size) or (index == last_index):
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)

                for model_name in do_models:
                    out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()

                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False
                    
                    if len(out_tens.shape) > 1:
                        logit1 = np.mean(out_tens, (1, 2))
                    else:
                        logit1 = out_tens

                    for ii, logit in zip(batch_id, logit1):
                        table.loc[ii, model_name] = logit

                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

            assert len(batch_id) == 0
        
    return table

def save_results(string_videos, video_name, input_csv, models, fusion_methods, threshold=0, just_soft_or_prob=False):
    """
    Save prediction results for video analysis to a CSV file.

    Parameters:
    -----------
    string_videos : str
        Identifier for the video batch being processed.
    video_name : str
        Name of the video being analyzed.
    input_csv : str
        Path to the CSV file containing model and fusion method predictions.
    models : list
        List of model names whose predictions are included in the analysis.
    fusion_methods : list
        List of fusion methods used for generating combined predictions.
    threshold : float, optional
        Decision threshold for classifying frames (default is 0).
    just_soft_or_prob : bool, optional
        If True, compute results only for 'fusion[soft_or_prob]' (default is False).

    Returns:
    --------
    str
        Path to the output CSV file containing the results.

    Raises:
    -------
    RuntimeError
        If an error occurs during processing, with details of the issue.
    """

    try:
        data = pd.read_csv(input_csv)
        output_csv = os.path.join(RESULTS_PATH, f'results_{string_videos}.csv')
        
        if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
            df = pd.read_csv(output_csv)
        else:
            columns = ['filename']
            if just_soft_or_prob:
                columns.append('prediction')
            else:
                columns += [f'prediction_{model}' for model in models] + \
                           [f'prediction_fusion[{method}]' for method in fusion_methods]
            df = pd.DataFrame(columns=columns)
        
        required_columns = [f'fusion[{method}]' for method in fusion_methods] + models
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the input CSV: {missing_columns}")
        
        if just_soft_or_prob:
            synthetic_count = (data['fusion[soft_or_prob]'] > threshold).sum()
            total_frames = len(data)
            new_row = {'filename': video_name, 'prediction': synthetic_count / total_frames}
        else:
            results = {'filename': video_name}
            for model in models + [f'fusion[{method}]' for method in fusion_methods]:
                synthetic_count = (data[model] > threshold).sum()
                total_frames = len(data)
                results[f'prediction_{model}'] = synthetic_count / total_frames
            new_row = results
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(output_csv, index=False)
        
        return output_csv

    except Exception as e:
        raise RuntimeError(f"Error analyzing video: {e}")

def process_single_folder(video):
    """
    Process videos from a specified folder by extracting frames, running model tests, 
    applying fusion methods, and saving results to CSV files.

    Parameters:
    -----------
    video : str
        Identifier to select the type of videos to process:
        '1' for Luma, '2' for Real, '3' for CogVideoX, and others for HunyuanVideo.

    Returns:
    --------
    None
        Results are saved as CSV files in the specified output directory.
    """
    
    dataset_path = os.path.join(parent_path, "dataset")
    video_paths= [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.mp4')]
    
    video_paths_luma = []
    video_paths_real = []
    video_paths_cogvideo = []
    video_paths_hunyuan = []
    
    for video_path in video_paths:
        filename = os.path.basename(video_path)
        if "Luma" in filename:
            video_paths_luma.append(video_path)
        elif "Real" in filename:
            video_paths_real.append(video_path)
        elif "CogVideoX" in filename:
            video_paths_cogvideo.append(video_path)
        elif "HunyuanVideo" in filename:
            video_paths_hunyuan.append(video_path)

    print("\n\n\nRunning tests on device: ", device, "\n\n\n")
    
    if video == '1':
        video_paths = video_paths_luma
        string_videos = "luma"
    elif video == '2':
        video_paths = video_paths_real
        string_videos = "real"
    elif video == '3':
        video_paths = video_paths_cogvideo
        string_videos = "cogvideo"
    else:
        video_paths = video_paths_hunyuan
        string_videos = "hunyuan"
    
    for video_path in video_paths:
    
        print("\n\n\nRunning tests on video: ", video_path, "\n\n\n")
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        csv_path = os.path.join(temp_dir, "input_images.csv")
        output_csv = os.path.join(RESULTS_PATH, f"{string_videos}/frames_results_{string_videos}_{video_name}.csv")
        
        print("Extracting frames from video...")
        frame_paths = extract_frames_from_video(video_path, temp_dir)
        
        generate_csv_from_frames(frame_paths, csv_path)
        print(f"Frames extracted and CSV generated: {csv_path}")

        table = running_tests(csv_path, weights_dir, models, device)

        for fusion_method in fusion_methods:
            table[f'fusion[{fusion_method}]'] = apply_fusion(table[models].values, fusion_method, axis=-1)

        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        table.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
        save_results(string_videos, video_name, output_csv, models, fusion_methods, JUST_SOFT_OR_PROB)

def process_all_dataset():
    """
    Process all MP4 videos in the dataset directory by extracting frames, 
    running model predictions, applying fusion methods, and saving results 
    with optical flow probabilities to CSV files.

    Returns:
    --------
    None
        Results and updated predictions are saved as CSV files in the output directory.
    """
    dataset_path = os.path.join(parent_path, "dataset")
    video_paths= [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.mp4')]

    for video_path in video_paths:
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        csv_path = os.path.join(temp_dir, "input_images.csv")
        output_csv = os.path.join(RESULTS_PATH, f"dataset/frames_results_{video_name}.csv")
        
        # CLIP prediction
        print("Extracting frames from video...")
        frame_paths = extract_frames_from_video(video_path, temp_dir)
        
        generate_csv_from_frames(frame_paths, csv_path)
        print(f"Frames extracted and CSV generated: {csv_path}")

        table = running_tests(csv_path, weights_dir, models, device)

        for fusion_method in fusion_methods:
            table[f'fusion[{fusion_method}]'] = apply_fusion(table[models].values, fusion_method, axis=-1)

        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        table.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
        output_csv_results = save_results("complete_dataset", video_name, output_csv, models, fusion_methods, JUST_SOFT_OR_PROB)            

        # OF prediction
        df = pd.read_csv(output_csv_results)
        if 'prediction_OF' not in df.columns:
            df['prediction_OF'] = None

        prediction = get_prob(video_path)

        df.loc[df['filename'] == video_name, 'prediction_OF'] = prediction

        df.to_csv(output_csv_results, index=False)

        print(f"Updated CSV with new OF prediction for {video_name}.")

                
if __name__ == "__main__":
    video = input("Which video do you want to test? \n 1. Luma \n 2. Real \n 3. CogVideoX-5B \n 4. Hunyuan \n 5. All dataset \n")
    while video not in ['1', '2', '3', '4', '5']:
        video = input("Invalid input. Please enter 1, 2, 3, 4 or 5: ")

    if video == '5':
        process_all_dataset()
    else:
        process_single_folder(video)
