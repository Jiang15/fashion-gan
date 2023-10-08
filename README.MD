# Fashion Generation through Controllable StyleGAN
<img src="images/PCA_result_example.jpg" alt="pca"/>
GANs for All: Supporting Fun and Intuitive Exploration of GAN Latent Spaces  
Author: Wei Jiang, Richard Lee Davis, Kevin Gonyop Kim, Pierre Dillenbourg  
https://proceedings.mlr.press/v176/jiang22a.html  
Abstract: We have developed a new tool that makes it possible for people with zero programming experience to intentionally and meaningfully explore the latent space of a GAN. We combine a number of methods from the literature into a single system that includes multiple functionalities: uploading and locating images in the latent space, image generation with text, visual style mixing, and intentional and intuitive latent space exploration. This tool was developed to provide a means for designers to explore the "design space" of their domains. Our goal was to create a system to support novices in gaining a more complete, expert understanding of their domain{’}s design space by lowering the barrier of entry to using deep generative models in creative practice.

## Dataset
We use [Zalando dataset](https://github.com/zalandoresearch/feidegger)
which can also be downloaded from [Google drive](https://drive.google.com/file/d/1vmBHGiqoICdaLaUqFUy6YiXMBLPuhTCy/view?usp=drive_link).
The dataset itself consists of 8732 high-resolution images, 
each depicting a dress from the available on the Zalando shop against a white-background.

## Train StyleGAN Model
Train fashion gan from scratch
```commandline
!python train.py --outdir "training_runs" --snap 20 --metrics "none" --data "data/square_256_imgs.zip"
```
If the resume the model from a checkpoint, we can --resume
```commandline
!python train.py --outdir "training_runs" --snap 20 --metrics "none" --data "data/square_256_imgs.zip" --resume "training_runs/00015-square_256_imgs-auto1-resumecustom/network-snapshot-000400.pkl"
```

## Explore Latent Space

## Finetune DALL-E Model

## Result

## WebApp


