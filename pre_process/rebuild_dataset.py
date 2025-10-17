import os
import shutil

class RebuildDataset:
    
    @staticmethod
    def rebuild(root: str, dest: str, test_list: str) -> None:
        test_files = []
        with open(test_list, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
        os.makedirs(dest, exist_ok=True)
        os.makedirs(dest + "/Train", exist_ok=True)
        os.makedirs(dest + "/Test", exist_ok=True)
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith('.mp4'):
                    continue
                src_path = os.path.join(dirpath, filename).replace("\\", "/")
                test = False
                for tf in test_files:
                    if tf in src_path:
                        test = True
                        break
                dst_path = dest + "/" + ("Test" if test else "Train") + "/" + filename
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)
    
    @staticmethod
    def rename(root: str, dest: str) -> None:
        os.makedirs(dest + "/Train", exist_ok=True)
        os.makedirs(dest + "/Test", exist_ok=True)
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith('.mp4'):
                    continue
                src_path = os.path.join(dirpath, filename).replace("\\", "/")
                if "Test" in dirpath:
                    dst_path = dest + "/Test/" + filename.replace("label_A", "Normal")
                else:
                    dst_path = dest + "/Train/" + filename.replace("label_A", "Normal")
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)
        