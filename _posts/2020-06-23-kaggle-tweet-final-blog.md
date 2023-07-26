---
layout: post
title: Tweet Sentiment Extraction - Final Blog
date: 2020-06-23
description: Extract support phrases for sentiment labels
tags: nlp data-science
categories: kaggle
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

This is final blog for this NLP competition. We will discuss some caveats
to move up through the leaderboard. We used the RoBERTa model in the midway blog for the infences. In this blog, we will discuss sentiment-specific predictions, noise of the data and post-processing tricks to improve the prediction scores.

## Sentiment-specific Jaccard score

If we breakdown the average jaccard scores based on the sentiment, the average Jaccard values of the three sentiments are:

- Positive: 0.581
- Negative: 0.590
- Neutral: 0.976

Many tweets with positive and negative sentiment have a jaccard score of zero. Let us figure out the issues.

## The Noise in labels & The Magic
At a first glimpse, those results look pretty weird as the selected texts look like random noise which are not a subset of the full text. For instance, [cases](https://www.kaggle.com/code/debanga/what-the-no-ise) found by DEBANGA RAJ NEOG:

1. Missing a ```!```  <span style="color:orange">Damn! It ```hurts!!!```

2. Missing a ```.```  <span style="color:orange">It is ```stupid...```

3. Missing ```d``` in ```good```?  <span style="color:orange">LOL. It's not ```goo```

4. Missing ```ng``` in ```amazing```?  <span style="color:orange">Dude. It's not ```amazi``` at all!

It was found that the noise originated from the consecutive spaces in the data. This insight can be leveraged to match the *noisy* selected text using the predicted probabilities of start and end indices at the token level and an alignment post-processing, which is called *the Magic* for this competition. This technique was implemented by the 1st place solution [here](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159254), and found to be super helpful, which can increase the CV score by around 0.2. The implementation idea of *the Magic* is sketched at below.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-tweet/final-blog/the-magic.png" title="The Magic" class="img-fluid rounded z-depth-1" %}
  <figcaption>The pattern of noisy labels and how to leverage it.</figcaption>
</div>



## Post-processing tricks

I campe up with a postprocessing method below which consistently helps to improve the CV score by about 0.001--0.002. This post-processing comprises of two tricks. The first one is to have a back-up indices with the second highest probabilities for both start and end indices of tokens, which will be used when the start indice is larger than the end indice. The code for the first trick is below.
The second trick deals with the special characters using the `regex` package, as shown in the function `post_process`.

```python
a, a_bak= np.argsort(preds_start_avg[k,])[::-1][:2]
b, b_bak = np.argsort(preds_end_avg[k,])[::-1][:2]
if a>b:
    if a_bak <= b and a > b_bak:
        st = tokenizer.decode(enc.ids[a_bak-2:b-1])
    elif a_bak > b and a <= b_bak:
        st = tokenizer.decode(enc.ids[a-2:b_bak-1])
    elif a_bak <= b_bak:
        st = tokenizer.decode(enc.ids[a_bak-2:b_bak-1])
    else:
        count_abn_2 += 1
        st = full_text
```

```python
import re
def post_process(x):
    if x.startswith('.'):
        x = re.sub("([\.]+)", '.', x, 1)
    if len(x.split()) == 1:
        x = x.replace('!!!!', '!')
        x = x.replace('???', '?')
        if x.endswith('...'):
            x = x.replace('..', '.')
            x = x.replace('...', '.')
        return x
    else:
        return x
```

Moreover, I submitted results with the highest local CV score rather than the one with the highest public leaderboard score. Luckily I survived the huge shakeup in the end. I ended up with **a solo silver medal for this competition, ranking 90th place out of 2225 teams in total.**