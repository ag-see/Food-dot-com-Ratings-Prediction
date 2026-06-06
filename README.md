# Food.com Recipe Analysis:

**Author:** Adrianne See, Aarman Sachdev

---

# Introduction

Food.com hosts hundreds of thousands of user-submitted recipes covering cuisines from around the world. Alongside recipe metadata such as ingredients, preparation time, and nutritional information, users can leave ratings and reviews describing their experience with each recipe.

This project investigates the following question:

> **What characteristics are associated with five-star recipes on Food.com, and can recipe metadata be used to predict whether a recipe will receive a perfect rating?**

Understanding these relationships can benefit recipe creators, food platforms, and recommendation systems by identifying factors associated with highly rated recipes.

The analysis uses:

* **83,782 recipes**
* **731,927 user interactions**

Relevant variables include:

| Column                       | Description                     |
| ---------------------------- | ------------------------------- |
| avg_rating                   | Average user rating of a recipe |
| calories                     | Calories per serving            |
| n_ingredients                | Number of ingredients           |
| n_steps                      | Number of recipe steps          |
| minutes                      | Preparation time                |
| protein, sugar, sodium, etc. | Nutritional measurements        |
| tags                         | User-assigned recipe categories |
| ingredients                  | Ingredient list                 |
| submitted_year               | Year recipe was submitted       |

---

# Data Cleaning and Exploratory Data Analysis

Several preprocessing steps were necessary before analysis:

1. Merged recipe and interaction datasets.
2. Replaced invalid ratings of 0 with missing values.
3. Computed average recipe ratings.
4. Extracted nutritional features from the nutrition column.
5. Converted percentage daily values into decimal form.
6. Parsed ingredient and tag lists.
7. Extracted submission year.
8. Created a binary target variable identifying recipes with an average rating of exactly 5.

The final cleaned dataset contains **83,782 recipes and 22 features**.

## Univariate Analysis

<iframe
 src="assets/uni-calories.html"
 width="900"
 height="600"
 frameborder="0">
</iframe>

The distribution of recipe calories is heavily right-skewed. Most recipes contain fewer than 500 calories per serving, while a relatively small number of recipes contain extremely large calorie counts.

This suggests Food.com recipes are generally moderate in calorie content, with a small number of highly calorie-dense outliers.

## Bivariate Analysis

<iframe
 src="assets/bivariate-rating-calorie.html"
 width="900"
 height="600"
 frameborder="0">
</iframe>

The boxplot compares recipe ratings across low-calorie and high-calorie recipes using the median calorie count as a threshold.

Both groups exhibit very similar rating distributions. Calorie content alone appears to have limited predictive power for determining whether a recipe receives a perfect rating.

## Interesting Aggregates

Recipes were grouped by ingredient count and summarized using average nutritional values and ratings.

| Ingredients | Avg Calories | Avg Rating |
| ----------- | ------------ | ---------- |
| 1           | 714.65       | 4.86       |
| 5           | 347.80       | 4.65       |
| 10          | 449.82       | 4.61       |

Interestingly, recipes with very few ingredients tend to receive slightly higher ratings. This may reflect a preference for simplicity and convenience among Food.com users.

---

# Assessment of Missingness

## NMAR Analysis

The `description` column contains 70 missing values.

I believe this missingness is **NMAR (Not Missing At Random)** because users who choose not to provide a description may systematically differ from users who do. Missing descriptions may reflect lower engagement or effort during recipe submission, making the missingness dependent on the unobserved value itself.

Additional information such as contributor account age or posting history could potentially explain this missingness mechanism.

## Missingness Dependency

The `avg_rating` column contains 2,609 missing values, corresponding to recipes that have never been rated.

Permutation tests were performed to determine whether missingness depends on observed variables.

| Variable | P-value |
| -------- | ------- |
| minutes  | < 0.001 |
| n_steps  | < 0.001 |
| calories | < 0.001 |

Since all p-values are extremely small, we reject the hypothesis that rating missingness is independent of these variables.

<iframe
 src="assets/nmar-cooking.html"
 width="900"
 height="600"
 frameborder="0">
</iframe>

Recipes with missing ratings tend to require more preparation time and contain more calories, suggesting the missingness mechanism is more consistent with MAR than MCAR.

---

# Hypothesis Testing

I investigated whether higher-protein recipes are more likely to receive perfect ratings.

### Null Hypothesis

High-protein and low-protein recipes have the same probability of receiving a five-star rating.

### Alternative Hypothesis

High-protein recipes are more likely to receive a five-star rating.

### Test Statistic

Difference in proportions of five-star recipes.

### Significance Level

α = 0.05

### Results

* Observed Difference = -0.036
* P-value = 1.0

Since the p-value is substantially larger than 0.05, I fail to reject the null hypothesis.

There is no evidence that recipes with higher protein content are more likely to receive perfect ratings.

---

# Framing a Prediction Problem

The prediction task is:

> **Can we predict whether a recipe will achieve a perfect five-star average rating?**

This is a **binary classification problem**.

### Response Variable

`five_star`

* True if average rating equals 5.0
* False otherwise

### Evaluation Metric

**F1-score**

The dataset contains more five-star recipes than non-five-star recipes, making F1-score a better measure than accuracy because it balances precision and recall.

### Features Available at Prediction Time

Only information available when a recipe is submitted was used:

* Nutritional information
* Preparation time
* Number of ingredients
* Number of steps
* Tags
* Ingredient lists
* Submission year

User reviews and ratings were excluded because they occur after publication.

---

# Baseline Model

The baseline model was a Logistic Regression classifier using only quantitative features:

* calories
* minutes
* n_steps
* n_ingredients
* total_fat
* sugar
* sodium
* protein
* saturated_fat
* carbohydrates

All features were quantitative and standardized using `StandardScaler`.

### Performance

| Metric    | Score |
| --------- | ----- |
| F1        | 0.53  |
| Precision | 0.62  |
| Recall    | 0.46  |

While better than random guessing, the model struggled to identify highly rated recipes accurately.

---

# Final Model

The final model extended the baseline using both engineered and textual features.

### Added Features

#### submitted_year

Recipe popularity and user preferences may change over time. Including submission year allows the model to capture temporal trends.

#### ingredient_text

Ingredient lists were transformed into text documents and vectorized using TF-IDF.

This allows the model to learn whether specific ingredients are associated with highly rated recipes.

#### tag_text

Recipe tags were similarly converted into TF-IDF features.

Tags capture abstract recipe categories and user-defined descriptors that may correlate with recipe popularity.

### Model Selection

The final model used:

* Logistic Regression
* TF-IDF feature extraction
* SelectKBest feature selection
* GridSearchCV hyperparameter tuning

Best parameters:

```python
kbest__k = 300
model__C = 10
```

### Performance

| Metric    | Baseline | Final |
| --------- | -------- | ----- |
| F1        | 0.53     | 0.60  |
| Precision | 0.62     | 0.66  |
| Recall    | 0.46     | 0.56  |

The addition of ingredient and tag information substantially improved performance, indicating that textual recipe information contains meaningful signals about recipe quality.

---

# Fairness Analysis

To evaluate whether model performance differs across groups, recipes were separated into:

* Low-calorie recipes (≤ median calories)
* High-calorie recipes (> median calories)

### Evaluation Metric

Precision

### Null Hypothesis

The model is equally precise for low-calorie and high-calorie recipes. Any observed difference is due to random chance.

### Alternative Hypothesis

The model is less precise for high-calorie recipes.

### Method

A permutation test was performed using the difference in precision between groups as the test statistic.

### Conclusion

The observed difference in precision was small and not statistically significant.

As a result, there is insufficient evidence to conclude that the classifier performs substantially worse for either calorie group. The model appears to treat high-calorie and low-calorie recipes similarly with respect to predictive precision.
