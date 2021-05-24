import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union
from tqdm import tqdm
import models.r21d.transforms.rgb_transforms as T
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import Compose
from torchvision.io.video import read_video
from utils.utils import (action_on_extraction, form_list_from_user_input,
                         form_slices, reencode_video_with_diff_fps, show_predictions_on_dataset)
PRE_CENTRAL_CROP_SIZE = (128, 171)
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]
CENTRAL_CROP_MIN_SIDE_SIZE = 112
DEFAULT_R21D_STEP_SIZE = 16
DEFAULT_R21D_STACK_SIZE = 16
class ExtractR21D(nn.Module):
    
    def __init__(self,video_path,output_path,
                 step_size=None,stack_size=None):
        super(ExtractR21D,self).__init__()
        self.file_names_list = os.listdir(video_path)
        self.file_path_list = [video_path+file_name for file_name in self.file_names_list]
        self.step_size = step_size
        self.stack_size = stack_size
        self.on_extraction = 'save_numpy'
        if self.step_size is None:
            self.step_size = DEFAULT_R21D_STEP_SIZE
        if self.stack_size is None:
            self.stack_size = DEFAULT_R21D_STACK_SIZE
        self.transforms = Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize(PRE_CENTRAL_CROP_SIZE),
            T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
            T.CenterCrop((CENTRAL_CROP_MIN_SIDE_SIZE, CENTRAL_CROP_MIN_SIDE_SIZE))
        ])
        self.show_pred = False
        self.output_path = output_path
        self.extraction_fps = None
        self.feature_type = 'r21d_rgb'
    def forward(self,indices):
        device = indices.device
        model = r2plus1d_18(pretrained=True).to(device)
        model.eval()
        model_class = model.fc
        model.fc = torch.nn.Identity()
        for idx in indices:
            print(idx,' '+self.file_names_list[idx])
            if(os.path.exists(self.output_path+self.file_names_list[idx].strip('.mp4')+'.npy')):
                print(self.output_path+self.file_names_list[idx].strip('.mp4')+'.npy',' alread exists')
                continue
            feats_dict = self.extract(device, model, model_class, self.file_path_list[idx])
            action_on_extraction(feats_dict, self.file_path_list[idx], self.output_path, self.on_extraction)
    def extract(self, device: torch.device, model: torch.nn.Module, classifier: torch.nn.Module,
                video_path: Union[str, None] = None
                ) -> Dict[str, np.ndarray]:
        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # read a video
        rgb, audio, info = read_video(video_path, pts_unit='sec')
        # prepare data (first -- transform, then -- unsqueeze)
        rgb = self.transforms(rgb)
        rgb = rgb.unsqueeze(0)
        # slice the
        slices = form_slices(rgb.size(2), self.stack_size, self.step_size)

        vid_feats = []

        for stack_idx, (start_idx, end_idx) in enumerate(slices):
            # inference
            with torch.no_grad():
                output = model(rgb[:, :, start_idx:end_idx, :, :].to(device))
                vid_feats.extend(output.tolist())

                # show predicitons on kinetics dataset (might be useful for debugging)
                if self.show_pred:
                    logits = classifier(output)
                    print(f'{video_path} @ frames ({start_idx}, {end_idx})')
                    show_predictions_on_dataset(logits, 'kinetics')

        feats_dict = {
            self.feature_type: np.array(vid_feats),
        }

        return feats_dict
    
if __name__=='__main__':
    video_path = '/home/tione/notebook/dataset/videos/video_5k/train_5k/'
    output_path = '/home/tione/notebook/dataset/r21d/'
    extractor = ExtractR21D(video_path=video_path,output_path=output_path)
    indices = torch.arange(len(extractor.file_path_list)).to('cuda')
    extractor.to('cuda')
    extractor(indices)