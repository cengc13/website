---
layout: post
title: M5 Forecasting Accuracy - Midway Blog
date: 2020-06-28
description: Estimate the unit sales of Walmart retail goods
tags: time-series-forecasting data-science
categories: kaggle
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

One of the most common techniques for time series forecasting is feature engineering. Effective feature engineering can boost the performance of your models. In this blog, we will discuss a few feature engineering strategies useful for this challenge.

## Feature Engineering (FE)

Since we have a large number of data, we can use the simplest data types for each column to reduce the memory usage. For example, the following function `reduce_mem_uage` reduces the memory for `df`, which is a `pandas` dataframe.

```python
## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```
Other simple methods to see memory consumption are:

```python
import numpy as np
import os, psutil
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2)

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
```

Another way to save memory is to condense a sparse matrix using `scipy.sparse.csr_matrix` method. For example, if you have a sparse dataframe `df`, you can save memory by

```python
from scipy import sparse
df = sparse.csr_matrix(df)
```
Many of the feature engineering ideas are conceived by [Konstantin Yakovlev](https://www.kaggle.com/kyakovlev).

### Simple FE

We first discuss simple methods based on statistics of existing variables.
Specific methods include

- Basic aggregations such as taking the `max`, `min`, `std` and `mean`
- Min/max scaling
- Unique items to identify items that may depend on inflation
- Rolling aggregations using months or years as windows.
- "Momentum" of prices. Prices that are shifted by week, month or year.

In addition, we can merge event features and snap features, and we can also use some features from date. Combining all those features we arrive at a initial dataset after simple feature engineering, column names and data types of which are shown as below.

<div class="img-div" markdown="0" style="text-align:center">
  {% include figure.html path="/assets/img/kaggle-m5/midway-blog/simple-fe.png" title="Data summary after simple feature engineering" class="img-fluid rounded z-depth-1" %}
  <figcaption>A summary of data info with simple feature engineering.</figcaption>
</div>

### Lags features

We can also create lags features by shifting the values by dates. Note that we need to sort the data by date before using shifts. Also note that we need to aggregate the data values on `id` (item) level. You can apply rolling max/min/mean with different time windows to get more lags features.

### Custom features

Other methods to customize and select features use simple and fast models (e.g. `LightGBM`) along with feature selection methods based on permutation tests, dimensional reduction techniques such as principal component analysis (PCA), and mean/std target encoding.

Suppose `grid_df` is the dataframe after the initial featurization. An iterative mean/std featurization implementation is given below:

```python
for col in icols:
    print('Encoding', col)
    temp_df = grid_df[grid_df['d']<=(1913-28)] # to be sure we don't have leakage in our validation set

    temp_df = temp_df.groupby([col,'store_id']).agg({TARGET: ['std','mean']})
    joiner = '_'+col+'_encoding_'
    temp_df.columns = [joiner.join(col).strip() for col in temp_df.columns.values]
    temp_df = temp_df.reset_index()
    grid_df = grid_df.merge(temp_df, on=[col,'store_id'], how='left')
    del temp_df
```

When all the features are generated and selected, the new dataframe is saved to hard disk for training models in the next step. In the next blog, we will discuss the model architecture and tricks to improve the model performance.