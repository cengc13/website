---
layout: page
title: Machine learning competitions on Kaggle
description: A list of past kaggle competitions
img: assets/img/projects/kaggle-competitions/kaggle-logo.png
importance: 1
category: Data science
---

### **About Kaggle**

>Kaggle is a data science competition platform and online community of
data scientists and machine learning practitioners under Google LLC.

This place hosts website links for blogs documenting my learning journey for past Kaggle machine learning competitions.

---

### **Competitions**

#### Bengali.AI Handwritten Grapheme Classification
Computer vision for multi-class classification

<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/kaggle-competitions/kaggle-bengali-desc.png" title="Bengali.AI description" width=320 height=400 class="img-fluid rounded z-depth-1"%}
</div>

> ##### **Competition description**
>Challenge and dataset summary paper available at <a href="https://arxiv.org/abs/2010.00170">arXiv</a>.
>
>Bengali is the 5th most spoken language in the world with hundreds of million of speakers. It’s the official language of Bangladesh and the second most spoken language in India. Considering its reach, there’s significant business and educational interest in developing AI that can optically recognize images of the language handwritten. This challenge hopes to improve on approaches to Bengali recognition.
>
>Optical character recognition is particularly challenging for Bengali. While Bengali has 49 letters (to be more specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This means that there are many more graphemes, or the smallest units in a written language. The added complexity results in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).
>
>For this competition, you’re given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.


##### Competition blogs

- [Initial blog](https://cengc13.github.io/website/kaggle/2020/02/18/kaggle-bengali-start.html)
- [Midway blog](https://cengc13.github.io/website/kaggle/2020/02/25/kaggle-bengali-midway.html)
- [Final blog](https://cengc13.github.io/website/kaggle/2020/03/03/kaggle-bengali-final.html)

**I completed this competitions as the leader of Team "Zzz..." and we won a silver medal through this effort, ranking 90$$^{\rm{th}}$$ out of 2059 teams.**

---

#### Jigsaw Multilingual Toxic Comment Classification
Natural language processing for zero- and few-shot learning in multilingual language classification.

<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/kaggle-competitions/kaggle-jigsaw.png" title="Jigsaw logo" width=320 height=320 class="img-fluid rounded z-depth-1"%}
</div>



> ##### **Competition description**
>
>Toxicity of an online comment is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion. If these toxic contributions can be identified, we could have a safer, more collaborative internet.
>In this competition, we're taking advantage of Kaggle's new TPU support and challenging you to build multilingual models with English-only training data.
>
>Over the past year, toxicity models have seen impressive multilingual capabilities from the latest model innovations, including few- and zero-shot learning. We're excited to learn whether these results "translate" (pun intended!) to toxicity classification. Your training data will be the English data and your test data will be Wikipedia talk page comments in several different languages.
>
>*Disclaimer*: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.


##### Competition blogs

- [Initial blog](https://cengc13.github.io/website/kaggle/2020/04/12/kaggle-jigsaw-start-blog.html)
- [Midway blog](https://cengc13.github.io/website/kaggle/2020/04/26/kaggle-jigsaw-midway-blog.html)
- [Final blog](https://cengc13.github.io/website/kaggle/2020/05/08/kaggle-jigsaw-final-blog.html)

**I completed this competitions with a solo gold medal, ranking 5$$^{\rm{th}}$$ out of 1621 teams.**

---

#### Tweet Sentiment Extraction
Natural language processing to find the text segment that supports the tweet sentiment.

<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/kaggle-competitions/tweet.gif" title="Tweet sentiment extraction" width=320 height=320 class="img-fluid rounded z-depth-1"%}
</div>



> ##### **Competition description**
>
>"My ridiculous dog is amazing." [sentiment: positive]
>
>With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.
>
>In this competition we've extracted support phrases from Figure Eight's Data for Everyone platform. The dataset is titled Sentiment Analysis: Emotion in Text tweets with existing sentiment labels, used here under creative commons attribution 4.0. international license. Your objective in this competition is to construct a model that can do the same - look at the labeled sentiment for a given tweet and figure out what word or phrase best supports it.


##### Competition blogs

- [Initial blog](https://cengc13.github.io/website/kaggle/2020/06/01/kaggle-tweet-start-blog.html)
- [Midway blog](https://cengc13.github.io/website/kaggle/2020/06/09/kaggle-tweet-midway-blog.html)
- [Final blog](https://cengc13.github.io/website/kaggle/2020/06/23/kaggle-tweet-final-blog.html)

**I completed this competitions with a solo silver medal, ranking 90$$^{\rm{th}}$$ out of 2225 teams.**

---

#### M5 Forecasting - Accuracy
Complex time series forecasting to predict the point of scales based on historical observations.

<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/kaggle-competitions/m5-graphical-concept.png" title="M5 Accuracy" width=320 height=320 class="img-fluid rounded z-depth-1"%}
</div>



> ##### **Competition description**
>
> This is one of the two complementary competitions that together comprise the M5 forecasting challenge. Can you estimate, as precisely as possible, the point forecasts of the unit sales of various products sold in the USA by Walmart? If you are interested in estimating the uncertainty distribution of the realized values of the same series.
>
>In this competition, the fifth iteration, you will use hierarchical sales data from Walmart, the world’s largest company by revenue, to forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.
>
>If successful, your work will continue to advance the theory and practice of forecasting. The methods used can be applied in various business areas, such as setting up appropriate inventory or service levels. Through its business support and training, the MOFC will help distribute the tools and knowledge so others can achieve more accurate and better calibrated forecasts, reduce waste and be able to appreciate uncertainty and its risk implications.


##### Competition blogs

- [Initial blog](https://cengc13.github.io/website/kaggle/2020/06/25/kaggle-m5-accuracy-start-blog.html)
- [Midway blog](https://cengc13.github.io/website/kaggle/2020/06/28/kaggle-m5-accuracy-midway-blog.html)
- [Final blog](https://cengc13.github.io/website/kaggle/2020/07/05/kaggle-m5-accuracy-final-blog.html)

**I finished this competition with a rank of 645/5558 teams.**