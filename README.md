# TDPDNE
A StyleGAN2 model to make AI generated dicks

Website: [https://thisdickpicdoesnotexist.com/](https://thisdickpicdoesnotexist.com/)

Demo Notebook: [Google Colab](https://colab.research.google.com/drive/1DoCxr2pYlxCRv6RmITtFWahVXsbTexYp?usp=sharing)

This project was based off of [https://thispersondoesnotexist.com/](https://thispersondoesnotexist.com/) but with dicks.

StyleGAN2 Paper: [https://arxiv.org/pdf/1912.04958.pdf](https://arxiv.org/pdf/1912.04958.pdf)

StyleGAN2 Repo: [https://github.com/NVlabs/stylegan2](https://github.com/NVlabs/stylegan2)

## Model Details

42273 dick pics were scraped from Reddit that were posted on these subreddits

* r/penis
* r/cock
* r/dicks
* r/averagepenis
* r/MassiveCock
* r/tinydick

The scraping procedure was done by first [downloading all submissions](https://files.pushshift.io/reddit/submissions/) from the year 2018. These submissions were then filtered down to image submissions in the above subreddits. Since pushshift only stores the image URLs, these images were then fetched from reddit using the stored URL.

A model was created with these images, but the model suffered from mode collapse. The solution was to train a custom Mask-RCNN model to segment the penis. With this segmentation, PCA was used to find the tilt, then rotate so the penis was vertical (in `Image Processor/align_images.py`). However this results in some penises being upside down. A possible improvement could be training another Mask R-CNN to detect the head of the penis and make sure that is always at the top half of the image.

The training was done using a TPU v3-8 trained for ~9 days (25000 KImg). Gamma was started at 100 and decreased by 25 each 10,000 KImg. The resulting model still suffers from some mode collapse as the generated dicks seen on [https://thisdickpicdoesnotexist.com/](https://thisdickpicdoesnotexist.com/) are lacking of the African American variety. This was found to be surprising as there were many coloured dicks in the dataset.

The generated dicks on [https://thisdickpicdoesnotexist.com/](https://thisdickpicdoesnotexist.com/) used a truncation_psi of 0.7.

[Email me](mailto:hello@thisdickpicdoesnotexist.com) if you have any questions.

## How to use

### Download the dataset

1. [Here]()

2. Unzip the tarball and place in the root directory of the repo

2. Tell all your friends you have more dick pics on your computer than them

### Run The Image Preprocessor

1. Train the custom Mask R-CNN model using

 `Image Processor/Dick_Pic_Mask-RCNN_Trainer.ipynb`

2. Align the dataset and resize using

 `Image Processor/align_images.py`

### Get a Google Compute Platform TPU instance

1. Apply for [TFRC] (https://www.tensorflow.org/tfrc) if you haven't already

2. Start the instance

3. Install python 3.7

4. Set the right environment variables

        export NOISY=0
        export DEBUG=0
        export LABEL_SIZE=0
        export MODEL_DIR=gs://your-gcp-bucket/model
        export BATCH_PER=4
        export BATCH_SIZE=32
        export RESOLUTION=512

### Train the Dick-GAN

1. Convert the images to TFRecords using

`stylegan2-tpu/dataset_tool.py create_from_images ~/datasets/aligned_images_tfrecords_dir ~/aligned_images`

2. Train the model

`stylegan2-tpu/run_training.py --result-dir=gs://your-gcp-bucket/model --data-dir=dataset --dataset=aligned_images --config=config-f --num-gpus=8 --mirror-augment=true`

### Generate Dicks

1. Run

`stylegan2-tpu/generate_images_tpu.py --model_dir=gs://your-gcp-bucket/model --save_dir=generated_fakes --truncation_psi=0.7 --num_samples=10`