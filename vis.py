import torch
import glob
import numpy as np
import os
import subprocess

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from models.model import generate_model
from learner import Learner
from PIL import Image

import numpy as np
import torch
import time
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
import argparse

parser = argparse.ArgumentParser(description='Video Anomaly Detection')
parser.add_argument('--n', default='', type=str, help='file name')
args = parser.parse_args()

class ToTensor(object):
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(self.norm_value)

        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        
        if pic.mode == 'YCbCr':
            nchannel = 3
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass

#############################################################
#                        MAIN CODE                          #
#############################################################


model = generate_model()  # feature extractor
classifier = Learner().cuda()  # classifier

checkpoint = torch.load('./weight/RGB_Kinetics_16f.pth')
model.load_state_dict(checkpoint['state_dict'])
checkpoint = torch.load('./weight/ckpt.pth')
classifier.load_state_dict(checkpoint['net'])

model.eval()
classifier.eval()

path = args.n + '/*'
save_path = args.n + '_result'
img = glob.glob(path)
img.sort()

# Create a directory to save the plots
plot_save_dir = args.n + '_result_plot'
if not os.path.exists(plot_save_dir):
    os.mkdir(plot_save_dir)

# Number of frames
total_frames = len(img)
inputs = torch.Tensor(1, 3, 16, 240, 320)
x_time = list(range(total_frames))
y_pred = [0] * total_frames  # Placeholder for predictions for all frames

os.system('cls' if os.name == 'nt' else 'clear')

for num, i in enumerate(img):
    
    FPS = 0.0
    out_str = 0.0
    
    if num < 16:
        inputs[:, :, num, :, :] = ToTensor(1)(Image.open(i))
        cv_img = cv2.imread(i)
        h, w, _ = cv_img.shape
        cv_img = cv2.putText(cv_img, 'FPS : 0.0, Pred : 0.0', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 240), 2)
    else:
        inputs[:, :, :15, :, :] = inputs[:, :, 1:, :, :]
        inputs[:, :, 15, :, :] = ToTensor(1)(Image.open(i))
        inputs = inputs.cuda()
        start = time.time()
        output, feature = model(inputs)
        feature = F.normalize(feature, p=2, dim=1)
        out = classifier(feature)
        y_pred[num] = out.item()
        end = time.time()
        FPS = str(1 / (end - start))[:5]
        out_str = str(out.item())[:5]

        cv_img = cv2.imread(i)
        cv_img = cv2.putText(cv_img, 'FPS :' + FPS + ' Pred :' + out_str, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 240), 2)
        if out.item() > 0.4:
            cv_img = cv2.rectangle(cv_img, (0, 0), (w, h), (0, 0, 255), 3)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    img_save_path = './' + save_path + '/' + os.path.basename(i)
    cv2.imwrite(img_save_path, cv_img)

    # Save the graph at the current frame
    plt.plot(x_time[:num + 1], y_pred[:num + 1], color="blue")  # Plot up to the current frame
    plt.xlim(0, total_frames)  # Set x-axis to the total number of frames
    plt.ylim(-0.1, 1)  # Assuming predictions are in the range [0, 1]
    plt.xlabel("Frame")
    plt.ylabel("Prediction")

    plot_file_path = os.path.join(plot_save_dir, f'{num:05}.jpg')
    plt.savefig(plot_file_path, dpi=300)
    plt.cla()  # Clear the plot for the next frame
    
    print(f'Frame {num}/{total_frames} : FPS {FPS}, Prediction {out_str}')

# Final video generation
os.system('ffmpeg -i "%s" "%s"' % (save_path + '/%05d.jpg', save_path + '.mp4'))

# Final plot of the entire graph
plt.plot(x_time, y_pred, color="blue")
plt.xlim(0, total_frames)
plt.ylim(-0.1, 1)
plt.xlabel("Frame")
plt.ylabel("Prediction")
plt.savefig(save_path + '.png', dpi=300)
plt.cla()

os.system('ffmpeg -i "%s" "%s"' % (plot_save_dir + '/%05d.jpg', plot_save_dir + '.mp4'))
