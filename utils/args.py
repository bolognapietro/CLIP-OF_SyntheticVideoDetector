import os

class Args():
    def __init__(self, path_input, small=False, mixed_precision=False, alternate_corr=False, use_cpu=True, aug_norm=True, dropout=True):
        self.model = "raft_model/raft-things.pth"
        self.path = path_input
        file_name = os.path.basename(path_input)
        self.folder_original_path = f"frame/{file_name}"
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr
        self.folder_optical_flow_path = f"optical_result/{file_name}"
        self.model_optical_flow_path = "checkpoints/optical.pth"
        self.use_cpu = use_cpu
        self.aug_norm = aug_norm
        self.dropout = dropout
        self.arch = "resnet50"
        pass
