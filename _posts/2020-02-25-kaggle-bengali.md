---
layout: post
title: Bengali.AI Handwritten Grapheme Classification&#150Midway Blog
date: 2020-02-25
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

## The model (densenet121)

In this [Kaggle competition](https://www.kaggle.com/c/bengaliai-cv19), we aim to develop a convolutional neural network (CNN) model to classify the three constituent components of Bengali handwritten characters, including grapheme root, vowel diacritics, and consonant diacritics. Identifying characters by optical recognition is challenging since each Bengali has 11 vowels and 38 consonants in its alphabet, and there are 10 potential diacritics. As a result, a large number of graphemes (the smallest units in a written language) exist, and this quickly adds up to more than 10,000 different grapheme variations. This work by Team **Zzz..** lives on [github](https://github.com/cengc13/Bengali_Kaggle).

## Overview of the data sets

### Parquet Files

The data sets are saved in the format of parquet files, which contain image IDs and the corresponding flattened 137 x 236 grayscale images. Each feature corresponds to a pixel of the image. The pixel values are between 0 and 255.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/parquet.png" title="parquet header" class="img-fluid rounded z-depth-1" %}
  <figcaption>Example parquet data for image pixels</figcaption>
</div>


### Training set

The training set contains image IDs from the parquet files and the 3 components of the corresponding graphemes, and there are 200,840 images in the training set. Note that the input is the handwritten image (the last column), while the output should be the classes for the corresponding three constituent components.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/training.png" title="training data header" class="img-fluid rounded z-depth-1" %}
  <figcaption>Example training data</figcaption>
</div>


### Test Set

The testing images consist of images whose constituent components are listed in independent rows.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/test.png" title="test data header" class="img-fluid rounded z-depth-1" %}
  <figcaption>Example test data</figcaption>
</div>

### Class Map

The class-map contains grapheme component types and labels, and it maps the class labels to the actual Bengali grapheme components.


<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/class-map.png" title="class map header" class="img-fluid rounded z-depth-1" %}
  <figcaption>Example class-map data</figcaption>
</div>

## Submission Format
The sample submission file has two columns---one column is the row ID from the test set which consists of the test index number and the component in a grapheme and the prediction.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/example-submission.png" title="Example submission header" class="img-fluid rounded z-depth-1" %}
  <figcaption>Example submission data</figcaption>
</div>


## Exploratory Data Analysis (EDA)


### Pixel distribution

The original pixel distribution is shown below, and it will be later used to compare with the pixel distributions after image crop and resize.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/pixel-dist.png" title="Pixel distribution of training images" class="img-fluid rounded z-depth-1" %}
  <figcaption>Pixel distributions of training images</figcaption>
</div>


### Class frequency analysis

#### Top 20 grapheme roots


Top 20 grapheme roots and their percentages in the training set are shown in the below figure. Those grapheme roots are approximately evenly distributed.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/grapheme-root.png" title="Grapheme roots" class="img-fluid rounded z-depth-1" %}
  <figcaption>Frequency of top 20 grapheme roots</figcaption>
</div>

#### Vowel diacritics

The counts of vowel diacritics are shown in the figure below. The distribution is not balanced, and they concentrate on Class 0, 1, 7, and 2.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/vowel-diacritic.png" title="Vowel diacritic" class="img-fluid rounded z-depth-1" %}
  <figcaption>Frequency of vowel diacritic</figcaption>
</div>

#### Consonant diacritics

For consonant diacritics, the distribution is not balanced either, with more than 60% being Class 0.


<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/consonant-diacritic.png" title="consonant diacritic" class="img-fluid rounded z-depth-1" %}
  <figcaption>Frequency of consonant diacritic</figcaption>
</div>

### Inspecting training images

#### Some randomly sampled images
Below is 25 example handwritten grapheme randomly chosen fro the training images.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/samples.png" title="Sample training images" class="img-fluid rounded z-depth-1" %}
  <figcaption>Randomly sampled example images</figcaption>
</div>

#### Writing variety

In the below it shows images of the same grapheme. Note that the handwriting of the same grapheme varies a lot.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/writing-variety.png" title="Writing variety" class="img-fluid rounded z-depth-1" %}
  <figcaption>Sixteen images of the same grapheme. Grapheme root, vowel diacritic and consonant diacritic are indexed 72, 1, 1, respectively.</figcaption>
</div>

## Data preprocessing

The images are standardized by cropping and resizing using methods implemented in the [OpenCV](https://github.com/opencv/opencv) package.
The method finds the contour of the figure and resize the image based on the size of the contour. In the following, we show the eight images after preprocessing and corresponding pixel distributions. The figures after processing look normal, and the pixel distribution with proprocessing is close to the one without preprocessing, implying the reliability of the method used.


<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/preprocessing.png" title="Preprocessed images" class="img-fluid rounded z-depth-1" %}
  <figcaption>Example handwritten grapheme after preprocessing</figcaption>
</div>

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-bengali/start-blog/pixel-dist-with-preprocessing.png" title="Pixel distribution after preprocessing" class="img-fluid rounded z-depth-1" %}
  <figcaption>Pixel distribution after preprocessing</figcaption>
</div>




