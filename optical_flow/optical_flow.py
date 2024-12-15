import os
import cv2
import sys
import torch
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from core.args import Args
from core.utils import flow_viz
from core.utils.utils import InputPadder
from optical_flow.core.raft_OF import RAFT
from core.utils1.utils import get_network, str2bool, to_cuda

DEVICE = 'cuda'

def load_image(imfile):
    """
    Loads an image from the specified file, converts it to a PyTorch tensor, 
    and moves it to the specified device.

    Parameters:
    -----------
    imfile : str
        Path to the image file.

    Returns:
    --------
    torch.Tensor
        A 4D tensor (1 x C x H x W) representing the image, moved to the specified device.
    """

    # Load the image as a NumPy array and convert it to uint8
    img = np.array(Image.open(imfile), dtype=np.uint8)

    # Convert the NumPy array to a PyTorch tensor and adjust dimensions
    img = torch.from_numpy(img).permute(2, 0, 1).float()

    # Add a batch dimension and move to the specified device
    return img.unsqueeze(0).to(DEVICE)

def viz(img, flo, folder_optical_flow_path, imfile1):
    """
    Saves a visualization of optical flow as an RGB image.

    Parameters:
    -----------
    img : torch.Tensor
        The input image tensor (1 x C x H x W).
    flo : torch.Tensor
        The optical flow tensor (1 x 2 x H x W).
    folder_optical_flow_path : str
        Path to the folder where the optical flow visualization will be saved.
    imfile1 : str
        Path to the input image file (used to derive the output filename).

    Returns:
    --------
    None
        Saves the optical flow visualization as an image.
    """
    
    # Convert tensors to NumPy arrays
    img = img[0].permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, C)
    flo = flo[0].permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, 2)

    # Map optical flow to an RGB image
    flo = flow_viz.flow_to_image(flo)

    # Create output path for the visualization
    filename = os.path.basename(imfile1)  # Extract the file name from the input path
    output_path = os.path.join(folder_optical_flow_path, filename.strip())

    # Save the flow visualization as an image
    cv2.imwrite(output_path, flo)

def video_to_frames(video_path, output_folder):
    """
    Extracts frames from a video and saves them to the specified folder.

    Parameters:
    -----------
    video_path : str
        Path to the input video file.
    output_folder : str
        Path to the folder where extracted frames will be saved.

    Returns:
    --------
    list[str]
        Sorted list of paths to the saved frame images.
    """

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Read and save frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

    # Get and sort all extracted frame paths
    images = sorted(glob.glob(os.path.join(output_folder, '*.png')) +
                    glob.glob(os.path.join(output_folder, '*.jpg')))
    
    return images


def OF_gen(args):
    """
    Generates and saves optical flow visualizations from a video.

    Parameters:
    -----------
    args : object
        Argument object containing paths to the video, model, and output directories.

    Returns:
    --------
    None
        Processes video frames, computes optical flow using a RAFT model, 
        and saves visualized results to the specified directory.
    """

    # Load the RAFT model
    print(f"Loading model on {DEVICE}")
    model = torch.nn.DataParallel(RAFT(args))
    print(args.model)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.module
    model.to(DEVICE)
    model.eval()

    # Ensure output directory exists
    os.makedirs(args.folder_optical_flow_path, exist_ok=True)

    # Process video frames and compute optical flow
    with torch.no_grad():
        print(f"----------------------------Processing video {args.path}----------------------------")
        
        # Extract frames from the video
        images = video_to_frames(args.path, args.folder_original_path)
        images = natsorted(images)

        # Iterate through pairs of consecutive frames
        for imfile1, imfile2 in tqdm(
            zip(images[:-1], images[1:]), 
            desc="Generating Optical Flow", 
            total=len(images) - 1, 
            dynamic_ncols=True
        ):
            # Load images
            image1 = torch.tensor(load_image(imfile1)).to(DEVICE)
            image2 = torch.tensor(load_image(imfile2)).to(DEVICE)

            # Pad images for model input
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # Compute optical flow
            _, flow_up = model(image1, image2, iters=20, test_mode=True)

            # Save visualized optical flow
            viz(image1, flow_up, args.folder_optical_flow_path, imfile1)

def get_prob(path_input):
    """
    Computes the average optical flow probability for images in the specified folder.

    Parameters:
    -----------
    path_input : str
        Path to the input dataset or video frames directory.

    Returns:
    --------
    float
        Average optical flow probability across all processed images.
    """

    # Initialize arguments and generate optical flow
    args = Args(path_input)
    OF_gen(args)

    # Load the pretrained model
    model_op = get_network(args.arch)
    state_dict = torch.load(args.model_optical_flow_path, map_location="cpu")
    model_op.load_state_dict(state_dict.get("model", state_dict))
    model_op.eval()

    if not args.use_cpu:
        model_op.cuda()

    # Define image transformation pipeline
    transform_pipeline = transforms.Compose([
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
    ])

    # Get all image paths from the optical flow folder
    optical_images = sorted(
        glob.glob(os.path.join(args.folder_optical_flow_path, "*.jpg")) +
        glob.glob(os.path.join(args.folder_optical_flow_path, "*.png")) +
        glob.glob(os.path.join(args.folder_optical_flow_path, "*.JPEG"))
    )

    # Initialize variables to compute the average probability
    total_prob = 0.0
    image_count = len(optical_images)

    # Process each image and compute probabilities
    for img_path in tqdm(optical_images, desc="Processing Images", dynamic_ncols=True, disable=image_count <= 1):
        img = Image.open(img_path).convert("RGB")
        img = transform_pipeline(img)

        if args.aug_norm:
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Prepare input tensor
        img_tensor = img.unsqueeze(0)
        if not args.use_cpu:
            img_tensor = img_tensor.cuda()

        # Compute probability
        with torch.no_grad():
            prob = torch.sigmoid(model_op(img_tensor)).item()
            total_prob += prob

    # Compute and return the average probability
    avg_prob = total_prob / image_count if image_count > 0 else 0.0
    print("Optical Flow Probability:", avg_prob)
    return avg_prob
