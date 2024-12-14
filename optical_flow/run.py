import os
import re
import csv
import subprocess

#video_folder = "demo_video/results_cogvideox-5b"
video_folder = "demo_video/Dataset_resized/real/"
script_command = "python demo.py"
mop_checkpoint = "checkpoints/optical.pth"
mor_checkpoint = "checkpoints/original.pth"
optical_flow_base = "optical_result"

#video_files = [f for f in os.listdir(video_folder) if f.endswith("_Real_Fake.mp4")]
video_files = [f for f in os.listdir(video_folder) if f.endswith("_Real.mp4")]

def parse_output(output, video_file):
    original_prob_match = re.search(r"original prob ([0-9.]+)", output)
    optical_prob_match = re.search(r"optical prob ([0-9.]+)", output)
    predict_match = re.search(r"predict:([0-9.]+)", output)

    name = os.path.basename(video_file)
    original_prob = float(original_prob_match.group(1)) if original_prob_match else 0.0
    optical_prob = float(optical_prob_match.group(1)) if optical_prob_match else 0.0
    predict = float(predict_match.group(1)) if predict_match else 0.0

    return {"name": name, "original_prob": original_prob, "optical_prob": optical_prob, "predict": predict}

data_to_save = []

for index, video_file in enumerate(video_files):
    base_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(video_folder, video_file)
    folder_original_path = os.path.join(video_folder, base_name)
    folder_optical_flow_path = os.path.join(optical_flow_base, f"{index:06d}")

    os.makedirs(folder_original_path, exist_ok=True)
    os.makedirs(folder_optical_flow_path, exist_ok=True)

    command = (
        f"{script_command} --use_cpu "
        f"--path \"{video_path}\" "
        f"--folder_original_path \"{folder_original_path}\" "
        f"--folder_optical_flow_path \"{folder_optical_flow_path}\" "
        f"-mop \"{mop_checkpoint}\" "
        f"-mor \"{mor_checkpoint}\""
    )

    try:
        output = subprocess.check_output(command, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione del comando per {video_file}: {e}")
        continue

    if not output:
        print(f"Output vuoto per il file: {video_file}")
        continue

    parsed_data = parse_output(output, video_file)
    data_to_save.append([parsed_data["name"], parsed_data["original_prob"], parsed_data["optical_prob"], parsed_data["predict"]])
    print(f"Nome oggetto: {parsed_data['name']}")
    print(f"Probabilità originale: {parsed_data['original_prob']}")
    print(f"Probabilità ottica: {parsed_data['optical_prob']}")
    print(f"Predict: {parsed_data['predict']}\n")

csv_file = "output_results.csv"
header = ["name", "original_prob", "optical_prob", "predict"]

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data_to_save)

print(f"Dati salvati in {csv_file}")
