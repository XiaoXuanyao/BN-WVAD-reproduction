from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import os
import shutil
import numpy as np
from PIL import Image
from utils.debug import Debug

class PreProcess():

    @staticmethod
    def batch_process(dataset_path: str, output_path: str, size=(224, 224), snippet_len: int=30*3, snippet_cnt: int=200, **kwargs):
        tasks = []
        with ProcessPoolExecutor(max_workers=16) as executor:
            for dirpath, dirnames, filenames in os.walk(dataset_path):
                for filename in filenames:
                    if not filename.endswith('.mp4'):
                        continue
                    video_path = os.path.join(dirpath, filename).replace("\\", "/")
                    file_out_path = output_path + "/" + dirpath.replace("\\", "/").split("/")[-1]
                    tasks.append(executor.submit(PreProcess.process, video_path, file_out_path, size=size, snippet_len=snippet_len, snippet_cnt=snippet_cnt, **kwargs))
            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as e:
                    Debug.log("PreProcess", f"Error processing video: {e}")

    @staticmethod
    def get_metadata(video_path):
        video = cv2.VideoCapture(video_path)
        frames_cnt = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video.release()
        return {
            "frames_cnt": frames_cnt
        }

    @staticmethod
    def process(video_path: str, output_path: str, size=(224, 224), snippet_len=30*3, snippet_cnt=200, **kwargs):
        Debug.log("PreProcess", f"Processing video: {video_path}")
        metadata = PreProcess.get_metadata(video_path)
        frames_cnt = metadata["frames_cnt"]
        if frames_cnt < snippet_len * 2:
            return
        if kwargs.get("max_frames") and frames_cnt > kwargs["max_frames"]:
            return
        video = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(int(metadata["frames_cnt"])):
            ret, frame = video.read()
            frame = cv2.resize(frame, (224, 224))
            if not ret:
                raise ValueError("Error reading video frame.")
            frames.append(frame)
        frames = np.array(frames)
        video.release()
        Debug.log("PreProcess", f"Processing {snippet_cnt} snippets...")
        starts = PreProcess.gen_start_pos(frames_cnt, snippet_cnt=snippet_cnt, snippet_len=snippet_len)
        for i, start in enumerate(starts):
            snippet = PreProcess.get_snippet(frames, start, snippet_len=snippet_len)
            path = output_path + "/" + video_path.split("/")[-1].replace(".mp4", "")
            os.makedirs(path, exist_ok=True)
            PreProcess.save_video(snippet, f"{path}/snippet_{i}.mp4")

    @staticmethod
    def gen_start_pos(frames_cnt, snippet_cnt=200, snippet_len=30*3):
        starts = []
        for i in range(snippet_cnt):
            pos = int((frames_cnt - snippet_len) * i // (snippet_cnt - 1))
            starts.append(pos)
        return starts

    @staticmethod
    def get_snippet(video_arr: np.ndarray, start_pos, snippet_len=30*3, **kwargs):
        res = video_arr[start_pos:start_pos+snippet_len]
        sample_cnt = kwargs.get("sample_cnt", 2)
        res = np.array([e for i, e in enumerate(res) if i % sample_cnt == 0])
        return res
    
    @staticmethod
    def save_video(video_arr: np.ndarray, path):
        height, width, _ = video_arr[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
        for frame in video_arr:
            out.write(frame)
        out.release()


class SplitDataset():

    @staticmethod
    def split(dataset_path: str, output_path: str, val_ratio=0.3):
        files = os.listdir(dataset_path)
        idx = [i for i in range(len(files))]
        np.random.shuffle(idx)

        split_point = int(len(files) * (1 - val_ratio))
        train_idx = idx[:split_point]
        val_idx = idx[split_point:]
        os.makedirs(output_path + "/Train", exist_ok=True)
        os.makedirs(output_path + "/Val", exist_ok=True)
        for i in train_idx:
            shutil.copy(dataset_path + "/" + files[i], output_path + "/Train/" + files[i])
        for i in val_idx:
            shutil.copy(dataset_path + "/" + files[i], output_path + "/Val/" + files[i])