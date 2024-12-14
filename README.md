# This is our groups' image captioning project
There are 2 type of architecture used: CNN+LSTM and fine-tuned transformer model <space><space> 
## CNN+LSTM model
For CNN+LSTM: the neccesary files are stored in this google drive link:[Links to our model](https://drive.google.com/drive/folders/18v09YgWkQH5rCCGB8Plr1Fp-OlTLNkTo?fbclid=IwZXh0bgNhZW0CMTEAAR1ah52SodFestsqlaHxbEB4d2iKP2dgLleBcdxQ13bLWyEFMmWyYcKlin8_aem_sAZjlE29ETad_xsLIlCspA) <br>
You need to download those files and put them in the same directory to the project folder. The explaination of the files is uploaded in the above link.

## folder base_model
This folder contains the initial approach to image captioning of our group: using **resnet101** for encoding, Bahdanau Attention, greedy search or beam search for generating captions, LSTM for decoding.<br>
Inorder to test generating a caption for a picture: you can replace the picture directory **image_path** in your own computer in file **base_model_infer.py**. The output should be the picture with 2 kinds of generating: greedy search and beam search.<br>
File **bleu_score_check** is used to calculate bleu score, you need to install the test dataset, which is flickr8k. This is the link to download the test dataset:[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)<br>

## folder glove
This is our evaluation model: using resnet101 for encoding, LSTM for decoding, glove embedding, soft attention and beam search.<br>
The use is the same as the previous folder **base_model**, replace the image directory in the file **glove_model_infer.py**.
