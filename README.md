# pytorch-paper-replicating-exercises
These are the exercises from the project **replicating a machine learning research paper** and creating a Vision Transformer (ViT) from scratch using PyTorch [https://github.com/mrdbourke/pytorch-deep-learning/blob/main/08_pytorch_paper_replicating.ipynb]

We'll see how ViT, a state-of-the-art computer vision architecture, performs on a FoodVision Mini problem.

More specifically, we're going to be replicating the machine learning research paper [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929)  (ViT paper) with PyTorch.

    see 08_pytorch_paper_replicating_exercise.ipynb

## Tools used: Python, PyTorch, Google Colab, Gradio, HuggingFace

# Goals: 
Replicate a computer vision model from a research paper, and apply it to a project.

## 1. Problem
Use computer vision to classify images as sushi, steak or pizza.

## 2. Data
The data are images derived from the Food101 dataset.

## 3. Evaluation
Multi Class Log Loss between the predicted probability and the observed target. For each image in the test set, predict a probability for each of the different classes of images (sushi, steak or pizza)

## 4. Features
The data is a collection of images of sushi, steak and pizza.

## 5. Model
Replicate the machine learning research paper [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929)  (ViT paper) with PyTorch.

## 6. Deployment
In addition to completing the paper-replicating exercise, I created a FoodVision Gradio App using a pre-trained ViT model from torchvision and deployed it on Hugging Face.  I followed the steps in the class notebook, except deployed the VIT model instead of the EffNetB2 model he used. You can see a live running example of it here: https://huggingface.co/spaces/shlomoc/foodvision_mini_vit

    see 09_pytorch_model_deployment.ipynb

## 7. Gradio app
I also created a more general Gradio app to generate a caption for an image using the Blip Image Captioning Model from Salesforce.  You can see a live running example of it here: https://huggingface.co/spaces/shlomoc/blip-image-captioning

