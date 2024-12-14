import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted

print(sys.path)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from optical_flow.core.raft_OF import RAFT
from core.args import Args
from core.utils.utils import InputPadder
from core.utils1.utils import get_network, str2bool, to_cuda

DEVICE = 'cuda:1'
# DEVICE = 'cpu'  # Changed to 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, folder_optical_flow_path, imfile1):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # print(folder_optical_flow_path)
    parts = os.path.split(imfile1)
    content=parts[1]
    folder_optical_flow_path=folder_optical_flow_path+'/'+content.strip()
    #print(folder_optical_flow_path)
    cv2.imwrite(folder_optical_flow_path, flo)


def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()

    images = glob.glob(os.path.join(output_folder, '*.png')) + \
             glob.glob(os.path.join(output_folder, '*.jpg'))
    images = sorted(images)
    
    return images


def OF_gen(args):
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    print(f"Loading model on {DEVICE}")

    # Load the model checkpoint

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    model = model.module
    model.to(DEVICE)
    model.eval()

    if not os.path.exists(args.folder_optical_flow_path):
        os.makedirs(args.folder_optical_flow_path)

    with torch.no_grad():
        print(f"----------------------------Processing video {args.path}----------------------------")
        images = video_to_frames(args.path, args.folder_original_path)
        images = natsorted(images)

        # Wrap the loop with tqdm for progress tracking
        for imfile1, imfile2 in tqdm(zip(images[:-1], images[1:]), desc="Generating Optical Flow", total=len(images) - 1, dynamic_ncols=True):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            viz(image1, flow_up, args.folder_optical_flow_path, imfile1)


def get_prob(path_input):
    args = Args(path_input)

    OF_gen(args)

    model_op = get_network(args.arch)
    state_dict = torch.load(args.model_optical_flow_path, map_location="cpu")

    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_op.load_state_dict(state_dict)
    model_op.eval()
    if not args.use_cpu:
        model_op.cuda()

    trans = transforms.Compose(
        (
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
        )
    )

    original_subsubfolder_path = args.folder_original_path
    optical_subsubfolder_path = args.folder_optical_flow_path
                    
    # RGB frame detection
    original_file_list = sorted(
        glob.glob(os.path.join(original_subsubfolder_path, "*.jpg")) +
        glob.glob(os.path.join(original_subsubfolder_path, "*.png")) +
        glob.glob(os.path.join(original_subsubfolder_path, "*.JPEG"))
    )
    
    # Optical flow detection
    optical_file_list = sorted(
        glob.glob(os.path.join(optical_subsubfolder_path, "*.jpg")) +
        glob.glob(os.path.join(optical_subsubfolder_path, "*.png")) +
        glob.glob(os.path.join(optical_subsubfolder_path, "*.JPEG"))
    )

    optical_prob_sum = 0
    count = 0

    # Use tqdm with a description
    for img_path in tqdm(optical_file_list, desc="Processing Images", dynamic_ncols=True, disable=len(optical_file_list) <= 1):
        img = Image.open(img_path).convert("RGB")
        img = trans(img)
        if args.aug_norm:
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        in_tens = img.unsqueeze(0)
        if not args.use_cpu:
            in_tens = in_tens.cuda()

        with torch.no_grad():
            prob = torch.sigmoid(model_op(in_tens)).item()
            optical_prob_sum += prob
            count += 1

    optical_predict = optical_prob_sum / count
    print("optical prob", optical_predict)

    return optical_predict