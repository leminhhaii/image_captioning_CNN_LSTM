# Fine_tune_model
**This is the code to build fine_tuned model we used in our Deep Learning Project**
To test the model, we need to run:

1. Download the model from: [finetunemodel.pth](https://drive.google.com/file/d/1PriildZeOc9GIVHLfVJPfolMgONJhr0_/view?usp=sharing)
2. Download requirement.txt, infer.py in our respiratory
3. Run command line: pip install -r requirements.txt
4. Run command line: python infer.py --model_path /path/to/your/model.pth --image_path /path/to/your/image

**Brief explaination about the files**:
1. Model Fine-tune.ipynb: code to fine-tune the model
2. Model eval + test.ipynb: code to evaluate the model (using BLEU matrices and test new image with the model)
3. requirement.txt: enviromment setup for infer.py
4. infer.py: test the model using new images
5. [finetunemodel.pth](https://drive.google.com/file/d/1PriildZeOc9GIVHLfVJPfolMgONJhr0_/view?usp=sharing): link to our fine-tuned model.
