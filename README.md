# This is our groups' image captioning project
There are 2 type of architecture used: CNN+LSTM and fine-tuned transformer model <space><space> 

**Download requirements**
1. Download requirement.txt, infer.py in our respiratory
2. Run command line: pip install -r requirements.txt

## CNN+LSTM model
For CNN+LSTM: the neccesary files are stored in this google drive link: [Links to our model](https://drive.google.com/drive/folders/18v09YgWkQH5rCCGB8Plr1Fp-OlTLNkTo?fbclid=IwZXh0bgNhZW0CMTEAAR1ah52SodFestsqlaHxbEB4d2iKP2dgLleBcdxQ13bLWyEFMmWyYcKlin8_aem_sAZjlE29ETad_xsLIlCspA) <br>
You need to download those files and put them in the same directory to the project folder. The explaination of the files is uploaded in the above link.

### Base_model
This folder contains the initial approach to image captioning of our group: using **resnet101** for encoding, Bahdanau Attention, greedy search or beam search for generating captions, LSTM for decoding.<br>
Inorder to test generating a caption for a picture: you can replace the picture directory **image_path** in file **base_model_infer.py** and run this file. The output should be the picture with 2 kinds of generating: greedy search and beam search.<br>
File **bleu_score_check** is used to calculate bleu score, you need to install the test dataset, which is flickr8k. This is the link to download the test dataset: [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)<br>

### Glove
This is our evaluation model: using resnet101 for encoding, LSTM for decoding, glove embedding, soft attention and beam search.<br>
The use is the same as the previous folder **base_model**, replace the image directory in the file **glove_model_infer.py**.

### Inception_model
This model use pretrained inception model instead of resnet. To test to code, replace the image directory in the file **inception_infer.py**.

## Fine-tune model
To test the model, we need to run:

1. Download the model from: [finetunemodel.pth](https://drive.google.com/file/d/1PriildZeOc9GIVHLfVJPfolMgONJhr0_/view?usp=sharing)
2. Run command line: python Fine_tune_model-main/infer.py --model_path /path/to/your/model.pth --image_path /path/to/your/image

**Brief explaination about the files**:
1. Model Fine-tune.ipynb: code to fine-tune the model
2. Model eval + test.ipynb: code to evaluate the model (using BLEU matrices and test new image with the model)
3. requirement.txt: enviromment setup for infer.py
4. infer.py: test the model using new images
5. [finetunemodel.pth](https://drive.google.com/file/d/1PriildZeOc9GIVHLfVJPfolMgONJhr0_/view?usp=sharing): link to our fine-tuned model.


## folder gen_caption
This includes some generated captions with images of our models. The name of picture describe the model that generated that caption.
