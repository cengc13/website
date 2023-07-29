---
layout: post
title: M5 Forecasting Accuracy - Final Blog
date: 2020-07-05
description: Estimate the unit sales of Walmart retail goods
tags: time-series-forecasting data-science
categories: kaggle
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

This is the final blog documenting my learning experience for the M5 accuracy kaggle competition. In the previous blogs, we walked through the general information about this competition and elaborated many methods for featurization. Here, we will discuss the models, a magic trick to improve the model performance and import findings summarized by the competition host.

## LightGBM model

We carried feature engineering to obtain additional features in addition to the features already found in the original dataset. We introduced 9 additional features, including two lags features with 7 and 28 days of shigt, 7 and 28 days of rolling mean with respect to lags features, three date features including 'week', 'quarter' and `mday`.

Next we define the categorical feauters and columns that will not be used for training. We have
```python
cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df["sales"]
```

Followed by feature specification, we define the dataset for lightdbm models. Next, the lightgbm hyperparameters are shown below:

```python
params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
        'verbosity': 1,
        'num_iterations' : 2500,
}
```

Training is as simple as

```python
m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=100)
```

After which, the model can be saved directly:

```python
m_lgb.save_model("model.lgb")
```

In terms of the inference, we predict the next day sales recursively by **predicting on past predictions**.

## A magic trick

It was found by many participants that multiplying the prediction by certain scaling coefficients can somehow improve the model performance on the public leaderboard. The multipliers should depend on the hierarchical levels. In addition, a rolling factor for future predictions may also be helpful. By inspecting the public leaderboard performance, it was found a rolling factor larger than 1 is most effective. However, for the private leaderboard performance, it turns out that a less than 1 rolling factor works best.

## Key findings by the host

The key findings of the host is [published](https://www.sciencedirect.com/science/article/pii/S0169207021001874?via%3Dihub) on International Journal of Forecasting. We list those findings as follows:

- **Superior performance of machine learning methods**. Unlike the first three M-series competitions demonstrate the merit of simplicity of models. This competition fully proved the power of machine learning methods, suggesting that top ranked teams all used ML models and achieve superior solutions compared to the benchmark methods.

- **Value of combining**. The model performance can be improved by combining the results from different models, even relatively simple ones.

- **Value of "cross-learning"**. Cross-learning implies using a single model to capture patterns of different time series trends, which may appear opposite to the value of combining. However, we can still employ multiple models, that look at different parts of the total data. Actually all top 50-performing methods somehow used "cross-learning" to exploit all of the information in the data set.

- **Notable differences between the winning methods and benchmarks used for sales forecasting**. Although the winning teams demonstrated overall advantages of ML methods, the actual differences at low-level aggregation were much smaller. Also, one should note that the benchmark methods, say exponential smoothing, overperform the vast majority teams (about 92.5%). It suggests that standard conventional simple methods may still be useful in assisting decision making to support the operations of retail companies.

- **Beneficial effects of external adjustments**. As mentioned in the previous section, using multipliers at different levels can help to improve forecasting accuracy. Some of those adjustments are not completely based on meaningful rationale but instead on the analytical alignment of predictions on the lowest aggregation level with the those at the highest levels.

- **Value added by effective CV strategies**. For complex forecasting tasks like this competition, adopting effective CV strategies is critical to capture post-sample accuracy in an objective manner, to avoid overfitting and to mitigate uncertainty. Yet, various CV methods can be applied. Some important factors to be considered include the time period for validation, the size of the validation windows, how those windows will be updated, and criteria to rationalize the CV scores.

- **Importance of exogenous/explanatory variables**. Methods solely rely on the historical data patterns may sometimes fail to account for the effects of holidays, special days, promotions, prices and weather. It was observed that price-related features were significantly important for improving forecasting accuracy. Besides, importance of exogenous variables was substantiated by comparisons between the benchmarks in this competition.

For this competition, I ended up with 645 place out of 5558 teams.