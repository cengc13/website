---
layout: post
title: Bengali.AI Handwritten Grapheme Classification - Final Blog
date: 2020-03-03
description: Classify the components of handwritten Bengali
tags: computer-vision data-science multiclass-classification
categories: kaggle
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---
<!-- En-Dash         &ndash;    &#150;
Em-Dash         &mdash;    &#151;
Minus Symbol    &minus;    &#8722; -->

**Team: Zzz...**

**Members: Cheng Zeng, Zhi Wang, Peter Huang**

## Model evaluation

All the desnet 121 details already explained at Midway Blog.
With all the transformed training data, we fed them into the data generator and trained the model. The training history with 30 epochs was saved and visualized. In the following two plots using one dataset as an example is shown.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/final-blog/training-history.png" title="History of training and validation loss" class="img-fluid rounded z-depth-1" %}
  <figcaption>Training and validation loss versus training epochs</figcaption>
</div>

The first thing we can see is that the loss decreases gradually with the number of epochs, indicative of the absence of overfitting for this model. Another important feature we can tell is that the loss of accuracy almost reaches a plateau when the epoch is up to 30. It suggests that further training for more epochs may not necessarily improve the accuracy of the model.

## Inference and Submission

Inference and Submission
In the first step, we define some parameters, including the original image size and the target image size after preprocessing, the number of channels for input images and the batch dimension for batch submission (A `TestDataGenerator` is created for batch submission).

Then we create the submission file by predicting the three constituent components of a Grapheme word.

For the testing images, we merely resized the images to the target size without augmentation. After that, we loaded the two pre-trained models for prediction. We used two models rather than only one because it takes advantage of the idea of ensemble prediction, which indeed pushes the leaderboard score up by about 0.35%.

In the end, we save the prediction results into a file named `submission.csv`, as detailed in the competition rules.

## Approaches for model improvement

### Different augmentation methods

We tried to use more aggressive augmentation methods such as `cutout` to mitigate the overfitting issue. It adds improved regularization for the CNN model. It masks out random sections of input images during training. See below for some examples.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/final-blog/example-augmentation.png" title="Image augmentation examples" class="img-fluid rounded z-depth-1" %}
  <figcaption>Example augmentation methods</figcaption>
</div>

### Increasing resolution of resized images

This can increase the public LB score by as much as 1%, from around 0.95 to 0.96. The top figure indicates resized images with size 64$$\times$$64, and the bottom plot shows the resized images with size 128x$$\times$$128. With a larger input image size, it makes sense that the accuracy is increased since more information is kept. The figure below shows the comparision of four example handwritten grapheme images using 64$$\times$$64 and 128x$$\times$$128 resizing.

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/final-blog/64x64.png" title="64x64 resizing" class="img-fluid rounded z-depth-1"%}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0" style="top:0px">
        {% include figure.html path="assets/img/projects/final-blog/128x128.png" title="128x128 resizing" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Example images of 64$$\times$$64 resizing (Left) and 128x$$\times$$128 resizing (Right)
</div>

### Ensembling

The two single models using `Densenet121` architecture with 128$$\times$$128 input size give public leaderboard (LB) scores of 0.9620 and 0.9630. Those two models are only different in the `random_state` for training data splitting. If we combine both models, it can lead to a LB of 0.9657, about 0.3% increase.


### Hyperparameter tuning

Since the training overall datasets are computationally expensive, we only explored a limited region of the parameter space. We found that these methods do not change the final validation accuracy significantly. We finally used `kernel_size=(3,3)`, initial learning rate of 0.001 with the `ReduceLROnPlateau` scheduler, and `relu` activation function.

## The best model

Till now, the best model we have is the `Densenet121` with input image size of 128x128, using a combination of shiftscalerotate and cutout as image augmentation, it gives a LB score of 0.9630. We use two models for prediction and submission on Kaggle, and the LB score is 0.9657, slightly better than a single model. The kaggle entry for the best model is here [Kaggle entry](https://www.kaggle.com/cengc13/bengali-handwritten-grapheme-inference?scriptVersionId=29454598).


## Future directions

As we noted when the competition was closed, the number of unique handwritten graphemes (four thousand) is way less than the number of all graphemes (more than ten thousand). It indicates that some graphemes may not be observed in the training set. This probably explains the power of aggressive augmentation in this competition. In light of this analysis, we can use the generative adversarial network (GAN) to make unseen images to further improve the model performance.


**Update:** We won a silver medal in this competition, ranked 90$$^{\rm{th}}$$ place among 2059 teams :fireworks:.