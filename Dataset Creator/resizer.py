import cv2
import cv_tools
from pathlib import Path
from tqdm import tqdm

folder = Path('D:\Vue\TPPDNE_nuxt\static\stylegan2_fakes')
save_folder = Path('D:\Vue\TPPDNE_nuxt_copy\static\stylegan2_fakes_small_comp')

for f in tqdm(list(folder.iterdir())):
    image = cv2.imread(str(f))
    image_resized = cv_tools.resize(image, resize_to=200)
    filename = str(save_folder/f.stem)+'.jpg'
    cv2.imwrite(filename, image_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
