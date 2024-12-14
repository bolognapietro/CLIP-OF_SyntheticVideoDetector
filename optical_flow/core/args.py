import os

class Args():
    def __init__(self, path_input, small=False, mixed_precision=False, alternate_corr=False, use_cpu=True, aug_norm=True, dropout=True):
        # Get the directory where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Go one directory up and construct the path to 'raft_model/raft-things.pth'
        self.model = os.path.join(base_dir, os.path.pardir, "raft_model", "raft-things.pth")


        # Other paths
        self.path = path_input
        file_name = os.path.basename(path_input)
        self.folder_original_path = os.path.join(base_dir, os.path.pardir, "frame", file_name)
        self.folder_optical_flow_path = os.path.join(base_dir, os.path.pardir, "optical_result", file_name)
        self.model_optical_flow_path = os.path.join(base_dir, os.path.pardir, "checkpoints", "optical.pth")

        # Flags and other parameters
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr
        self.use_cpu = use_cpu
        self.aug_norm = aug_norm
        self.dropout = dropout
        self.arch = "resnet50"
