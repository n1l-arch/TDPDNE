import cv2
from pathlib import Path
import numpy as np
import cv_tools
from multiprocessing import Pool
from typing import Tuple
from tqdm import tqdm


def clean(image_and_savename: Tuple[np.ndarray, str], side_length=512) -> np.ndarray:
    try:
        image_name = image_and_savename[0]
        save_name = image_and_savename[1]

        if save_name.exists():
            return

        image = cv2.imread(str(image_name))

        if image.shape == (81, 161, 3):
            return

        padded_image = cv_tools.pad_to_square(image)

        resize_dim = (side_length, side_length)

        if padded_image.shape[0] < side_length:
            interpolation = cv2.INTER_CUBIC
            padded_image = cv2.resize(padded_image, resize_dim,
                                      interpolation=interpolation)
        elif padded_image.shape[0] > side_length:
            interpolation = cv2.INTER_AREA
            padded_image = cv2.resize(padded_image, resize_dim,
                                      interpolation=interpolation)

        cv2.imwrite(str(save_name), padded_image)
    except AttributeError:
        pass


if __name__ == '__main__':
    data_folder = Path('images')
    cleaned_folder = Path('cleaned_images_512')
    image_files = list(data_folder.iterdir())
    save_names = [cleaned_folder/f.name for f in image_files]
    images_and_names = list(zip(image_files, save_names))
    # for cleaned_image in cleaned_folder.iterdir():
    #     save_name = Path(data_folder/cleaned_image.name)
    #     if save_name in image_files:
    #         image_files.remove(save_name)

    with Pool(10) as p:
        for cleaned_image in tqdm(p.imap(clean, images_and_names), total=len(images_and_names)):
            pass
    # for image_file in tqdm(image_files):
    #     save_name = Path(cleaned_folder/image_file.name)
    #     if not save_name.exists():
    #         image = cv2.imread(str(image_file))
    #         cleaned_image = clean(image)
            # cv2.imwrite(str(save_name), cleaned_image)
            # break
