import os
import multiprocessing as mp
import numpy as np
from rembg import remove
from PIL import Image
from tqdm import tqdm

def remove_background(in_path, out_path):
    with open(in_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGBA')
            arr = np.array(img)
            arr[:, :, 3] = 255
            arr = remove(arr)
            # arr = remove(arr, alpha_matting=True, alpha_matting_foreground_threshold=240)
            img = Image.fromarray(arr, mode='RGBA')
            img.save(out_path)

if __name__ == "__main__":
    in_dir  = "data"
    out_dir = "rembg-data"
    
    pool = mp.Pool(8)

    for root, dirs, files in os.walk(in_dir):
        for name in tqdm(files):
            if not name.endswith('.png') and not name.endswith('.jpg'):
                continue
            in_path = os.path.join(root, name)
            rel_path = os.path.relpath(in_path, in_dir)
            out_path = os.path.join(out_dir, rel_path)
            out_path = os.path.splitext(out_path)[0] + '.png'
            out_dirname = os.path.dirname(out_path)
            if not os.path.exists(out_dirname):
                os.makedirs(out_dirname)
            pool.apply_async(remove_background, args=(in_path, out_path))

    pool.close()
    pool.join()