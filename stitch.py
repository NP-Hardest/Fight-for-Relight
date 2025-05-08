import cv2
import os
from natsort import natsorted

frame_folder = '/home/stud1/Nishant/PBR_relighting/outputs/vid'  

frame_files = [f for f in os.listdir(frame_folder) if f.endswith(('.png', '.jpg'))]
frame_files = natsorted(frame_files)

first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
height, width, _ = first_frame.shape

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

for filename in frame_files:
    frame_path = os.path.join(frame_folder, filename)
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()

