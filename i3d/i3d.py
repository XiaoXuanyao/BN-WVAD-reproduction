import torch
import os
import cv2
import numpy as np
import queue
import time
import threading
from pytorch_i3d.pytorch_i3d import InceptionI3d
from utils.debug import Debug
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Interface for I3D model
class I3DInterface():

    def __init__(self):
        self.inputs = queue.Queue()
        self.outputs = queue.Queue()
        self.finished = False
        self.done = False
        pass
    
    def batch_predict(self, dataset_path, output_path, snippet_len=30*3, batch_size=8):
        Debug.log("I3D", "Starting batch prediction...")

        threading.Thread(target=self.predict, args=(batch_size,), daemon=True).start()

        self.output_path = output_path + "/Train"
        self.files = []
        for file in os.listdir(dataset_path + "/Train"):
            self.files.append(dataset_path + "/Train/" + file)
        self.batch_load_inputs(snippet_len=snippet_len)
        
        while not self.done:
            time.sleep(0.1)
        self.done = False

        self.output_path = output_path + "/Test"
        self.files = []
        for file in os.listdir(dataset_path + "/Test"):
            self.files.append(dataset_path + "/Test/" + file)
        self.batch_load_inputs(snippet_len=snippet_len)
        
    
    def batch_load_inputs(self, snippet_len=30*3):
        tasks = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for video_folder in self.files:
                tasks.append(executor.submit(self.load_input, video_folder, snippet_len=snippet_len))
            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as e:
                    Debug.log("I3D", f"Error processing video: {e}")
    

    def load_input(self, video_folder, snippet_len=30*3):
        while self.inputs.qsize() >= 6:
            time.sleep(0.05)
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
        self.inputs.put((video_folder.split("/")[-1], snippets))


    def predict(self, batch_size):
        model = InceptionI3d(400, in_channels=3)
        model.load_state_dict(torch.load('pytorch_i3d/models/rgb_imagenet.pt'))
        model.eval().cuda().half()
        with torch.no_grad():
            while True:
                while self.inputs.empty():
                    if self.finished:
                        Debug.log("I3D", "Finished processing all videos.")
                        return
                    time.sleep(0.05)
                video_folder, snippets = self.inputs.get()
                all_outputs = []
                for i in range(0, len(snippets), batch_size):
                    batch_snippets = snippets[i:i+batch_size]
                    batch_np = np.array(batch_snippets, dtype=np.uint8)
                    batch_np = batch_np.astype(np.float16)
                    batch_tensor = torch.from_numpy(batch_np).permute(0, 4, 1, 2, 3)
                    batch_tensor = batch_tensor.cuda().half() / 255.0
                    outputs = model(batch_tensor)
                    all_outputs.append(outputs.cpu().numpy())
                all_outputs = np.concatenate(all_outputs, axis=0)
                output_file = os.path.join(self.output_path, video_folder + '.npy')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                np.save(output_file, all_outputs)
                Debug.log("I3D", f"Saved predictions to {output_file}")
                if self.inputs.qsize() == 0:
                    self.done = True