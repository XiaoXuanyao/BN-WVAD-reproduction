import torch
import os
import cv2
import numpy as np
import gc
from pytorch_i3d.pytorch_i3d import InceptionI3d
from utils.debug import Debug

# Interface for I3D model
class I3DInterface():

    def __init__(self):
        self.model = InceptionI3d()
        self.model.load_state_dict(torch.load("pytorch_i3d/models/rgb_imagenet.pt"))
        self.model.eval().half().cuda()
    
    def batch_predict(self, dataset_path, output_path, snippet_len=30*3, batch_size=8):
        Debug.log("I3D", "Starting batch prediction...")
        # for file in os.listdir(dataset_path + "/Train"):
        #     self.predict(dataset_path + "/Train/" + file, output_path + "/Train", snippet_len=snippet_len, batch_size=batch_size)
        for file in os.listdir(dataset_path + "/Test"):
            self.predict(dataset_path + "/Test/" + file, output_path + "/Test", snippet_len=snippet_len, batch_size=batch_size)

    def predict(self, video_folder, output_path, snippet_len=30*3, batch_size=8):
        Debug.log("I3D", f"Predicting video: {video_folder.split('/')[-1]}")
        os.makedirs(output_path, exist_ok=True)
        snippets = []
        for file in os.listdir(video_folder):
            if file.endswith(".mp4"):
                snippet = cv2.VideoCapture(os.path.join(video_folder, file))
                frames = []
                for _ in range(snippet_len):
                    ret, frame = snippet.read()
                    if not ret:
                        raise ValueError("Error reading video frame.")
                    frames.append(frame)
                snippets.append(np.array(frames))
                snippet.release()
        Debug.log("I3D", f"Move to GPU...")
        snippets = np.array(snippets, dtype=np.float16) / 255.0
        snippets = torch.from_numpy(snippets).half().permute(0, 4, 1, 2, 3)
        Debug.log("I3D", f"Predicting...")
        logits = []
        with torch.no_grad():
            for i in range(0, len(snippets), batch_size):
                torch.cuda.empty_cache()
                batch = snippets[i:i+batch_size].cuda()
                logits_0 = self.model(batch)
                logits.append(logits_0)
                Debug.log("I3D", f"Processed {i+len(batch)}/{len(snippets)} snippets", end="\r", type=1)
        logits = torch.cat(logits, dim=0)
        logits_np = logits.cpu().numpy()
        np.save(os.path.join(output_path, f"{video_folder.split('/')[-1]}.npy"), logits_np)
        del logits_np
        del snippets
        gc.collect()