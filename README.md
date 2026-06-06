# "What makes the Gold Standard?": Food.com Five Star Recipe Analysis and Prediction:

**Author:** Adrianne See, Aarman Sachdev

---
# Overview

This data science project explores the effectiveness of predictive models in understanding a 'Five Star Rating' for any given user recipe from Food.com.

# Introduction

Food.com is one of the largest food related websites on the Internet. Since 1999, it has served as a hub for home cooks, celebrities, chefs and companies to publish and share recipes from across the world. On a daily basis, The recipes are primarily rated through a five star system, and currently, over 500,000 user made recipes from around the world are accessible, including user feedback, reviews and images of other users making them at home.

Because of the social media aspect of the website, understanding what drives a 'Five Star Rating' becomes highly valuable in many ways. In particular, we wanted to investigate **What characteristics are associated with five-star recipes on Food.com, and can recipe metadata be used to predict whether a recipe will receive a perfect rating?*. Recipe creators, food platforms, and recommender systems can benefit from understanding meals that people enjoy. Depending on factor such as nutritional content, calorie density, and ease of creation, and ingredient makeup, general trends in food preferences can also be discovered, at the very least, in the online home cook community. Finally, any discoveries on tags tags, assigned by recipe creators, can assist in effectively categorizing and describing recipes, as well as reveal trends that may be otherwise hard to parse.

To perform this analysis, we analyzed a subset user recipes and user ratings from Food.com. Each recipe includes nutritional information, preparation details, and user-submitted ratings. We cleaned several columns and expanded their data (`calories`, `total_fat`, `submitted_year`) into separate columns, as well as extracted our derived feature, `five_star`, into its own target column to better capture our desired group. 

The first dataset we used, `recipe`, contains a subset (83,782) of all user submitted recipes uploaded to Food.com. Each entry represents a unique recipe submitted and viewable on the website, and contains the following features:

| Column Name | Description | 
|--------|-------------------|
| `name` | Recipe name |
| `id` | Recipe ID |
| `minutes` | Minutes to prepare recipe2 |
| `contributor_id`	 | User ID who submitted this recipe |
| `submitted` | Date recipe was submitted |
| `tags` | Food.com tags for recipe |
| `nutrition` | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for - “percentage of daily value” |
| `n_steps` | Number of steps in recipe |
| `steps` | Text for recipe steps, in order |
| `description` | User-provided description |

The second dataset we used, `ratings`, contains a subset (731,927) of all user submitted ratings uploaded to Food.com. Each entry represents a unique review for a unique recipe on the website, and contains the following features:

| Column Name | Description | 
|--------|-------------------|
| `user_id'` | User ID |
| `recipe_id` | Recipe ID |
| `date` | Date of interaction |
| `rating`	 | Rating given |
| `review` | Review text |

# Data Cleaning

While useful, the dataset is split into multiple parts. To further facilitate simple analysis and modeling, we will combine the information into one usable and analyzable dataset. To do so, the following transformations were performed, resulting in one homogenous `recipe` DataFrame.

1. **Merged recipes and interactions:** We performed a left merge on the `recipes` dataset with the `interactions` dataset, on `id` for recipes and `recipe_id` with interactions. This was done to associate each recipe with a rating column.

2. **Replaced 0 ratings with NaN:** Ratings of 0 were replaced with `np.nan`. After testing, a user account was unable to give a rating of 0 stars. This supports the conclusion that recipe ratings of zero stars are more likely due to data error or lack of user rating, rather than genuine ratings. Replacing 0's with `np.nan` prevents them from skewing the rating values when computing the mean rating per recipe.

3. **Computed average rating per recipe:** The mean rating for each recipe was calculated and assigned onto the `recipes` dataset as `avg_rating`.

4. **Parsed nutrition column:** The `nutrition` column stores `calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV)`, with the data looking like a list stored in string format. We converted these to actual lists and extracted individual nutritional values, creating the appropriate column names.

5. **Decimal transform of non caloric nutrition columns:** `total fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, and `carbohydrates` are all in Percentage Daily Value (PDV). For our analysis purposes, we found it useful to convert the percentages to decimals.

6. **Parsed tags column:** The `tags` column contains tags for the recipe, contained as a list in string format. These were converted into lists for future use. 

7. **Extracted Recipe Creation Year into `submitted_year`:** The year of recipe submission is present in the Dataset, but contaiend wihtin the submission date. This feature was extracted and put into `submitted_year` for better analysis and feature use.

8. **Create Target Column, ``five_star``:** The target prediction is the group of ratings that are equal to 5, or 5.0. A simple boolean mask was performed on `avg_rating` to produce the target column.

After the following Data Cleaning, `recipes`, now has 83782 entries and 22 features, describing each recipe with its appropriate `five_star` binary classifier column. Below is a table describing the resulting data types in each column of our dataset.

| Column | Data Type |
|----------|----------|
| name | object |
| id | int64 |
| minutes | int64 |
| contributor_id | int64 |
| submitted | object |
| tags | object |
| nutrition | object |
| n_steps | int64 |
| steps | object |
| description | object |
| ingredients | object |
| n_ingredients | int64 |
| avg_rating | float64 |
| calories | float64 |
| total_fat | float64 |
| sugar | float64 |
| sodium | float64 |
| protein | float64 |
| saturated_fat | float64 |
| carbohydrates | float64 |
| submitted_year | int32 |
| five_star | bool |

From here, special attention will be given to the following columns

- `minutes`
- `tags`
- `nutriton`, and its resulting columns `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrates`
- `submitted_year`

## Univariate Analysis

For a univariate analysis, we wanted to carefully analyze the distribution of caloric count in a recipe. The histogram below shows the distribution of calories across all recipes from our sample of Food.com.

<iframe
 src="assets/uni-calories.html"
 width="100%"
 height="600"
 frameborder="0">
</iframe>

A large amount of recipes (75.1%), contain strictly less than 500 calories, and the majority (94.2%) contain strictly less than 1000 calories. The graph is also right skewed, indicating that increasingly fewer recipes on Food.com have higher caloric count. This suggests that recipes on Food.com are generally modest to moderate in calorie content, with higher calorie recipes being outliers.

A table is provided below, containing more numeric summaries of the `calorie` column. This table is an augmented output of the useful `pd.DataFrame.describe()` function, which provided summary statistics of teh column.

| Statistic | Calories |
|------------|----------:|
| Count | 83,782 |
| Mean | 429.93 |
| Standard Deviation | 636.63 |
| 0th Percentile (Minimum) | 0.00 |
| 25th Percentile | 171.33 |
| 50th Percentile (Median) | 305.40 |
| 75th Percentile | 498.70 |
| 100th Percentile (Maximum) | 45,609.00 |

Of note is the difference of the mean and the median. The mean sits at 429.93 calories, above the median (305.4 calories) at a rate of 0.2 times the Standard Deviation. This indicates a right skew is present on the `calorie` column. 

## Bivariate Analysis

For a Bivariate Analysis, we wanted to compare the relationship between recipe calorie content and the likelihood of it to receive a five-star average rating. The bar plot below compares the proportion of recipes in each group (below median calories, above median calories) that achieved a five-star average rating.

<iframe
 src="assets/bivariate-rating-calorie.html"
 width="100%"
 height="600"
 frameborder="0">
</iframe>

With a proportional difference of ~0.01, both groups end up having a similar proportion of five-star recipes. This suggests that calorie content alone does not appear to be a strong predictor for determining recipe success on Food.com. This also indicate that more features may need to be included in future predictive models.

## Interesting Aggregates

For aggregation, we wanted to view how ingredient count affected columns like `calories`, `total_fat`, and `avg_rating`. The table below groups recipes by ingredient count and summarizes each group using the above metrics. As for the summary statistic, reflecting a pattern of skewedness being found in multiple metrics, both mean and median were included to check for further skews.


| n_ingredients | calories (mean) | calories (median) | total_fat (mean) | total_fat (median) | avg_rating (mean) | avg_rating (median) |
|--------------:|----------------:|------------------:|-----------------:|-------------------:|------------------:|--------------------:|
| 1  | 714.65 | 200.35 | 0.91 | 0.03 | 4.86 | 5.0 |
| 2  | 331.98 | 148.10 | 0.26 | 0.03 | 4.69 | 5.0 |
| 3  | 306.11 | 166.70 | 0.22 | 0.06 | 4.66 | 5.0 |
| 4  | 336.01 | 195.60 | 0.24 | 0.09 | 4.63 | 5.0 |
| 5  | 347.80 | 224.20 | 0.25 | 0.13 | 4.65 | 5.0 |
| 6  | 355.89 | 245.55 | 0.26 | 0.15 | 4.63 | 5.0 |
| 7  | 391.13 | 272.30 | 0.30 | 0.18 | 4.62 | 5.0 |
| 8  | 387.03 | 290.90 | 0.30 | 0.20 | 4.61 | 5.0 |
| 9  | 422.50 | 306.55 | 0.32 | 0.21 | 4.61 | 5.0 |
| 10 | 449.82 | 322.00 | 0.35 | 0.22 | 4.61 | 5.0 |

---

# Assessment of Missingness

## NMAR Analysis

The `description` column contains 70 missing values.

We believe this is likely **NMAR** (*Not Missing At Random*). A plauusible explanation we landed on was that as users who do not provide a description have behavioral patterns that are outside of the dataset. A hypothetical example would be users who put less effort into their recipe submissions may not necessarily write a descrioption. Because the probability of missigness in this situation may be related to the missing value itself, the mechanism in this scenario is very likely NMAR.

Additional information about the user itself could be helpful in determining between NMAR and **MAR** (*Missing At Random*). Because our current feature set does not tell us much about the user's behavioral patterns, especially within Food.com, desirable features such as activity metrics, previous submissions, or follower count, may explain why users might not provide recipe descriptions. Of note is that data like this does exist and is acessible on Food.com, albeit this does raise privacy concerns. If such data was accessible, then missignes could be re-evaluated further and possibly even be identified as MAR rather than NMAR.

## Missingness Dependency

We analyzed the missingness of `avg_rating`, which has 2,609 missing values. A recipe has a missing `avg_rating` if no users have rated it.

We tested whether the missingness of `avg_rating` depends on `calories`, `n_ingredients`, and `minutes` using permutation tests with the difference in means as our test statistic.

| Column | Observed Difference | P-value |
|--------|-------------------|---------|
| minutes | 117.342 | 0.0000 |
| n_steps | 1.493 | 0.0000 |
| calories | 87.859 | 0.0000 |

All three p-values are essentially 0, meaning we reject the null hypothesis that the missingness of `avg_rating` is independent of these columns. Recipes with missing ratings tend to have higher calories, more ingredients, and longer prep times. This suggests the missingness of `avg_rating` is **MAR** — dependent on other observed columns.

<iframe
 src="assets/nmar-cooking.html"
 width="100%"
 height="600"
 frameborder="0">
</iframe>

Recipes with missing ratings tend to require more preparation time and contain more calories, suggesting the missingness mechanism is more consistent with MAR than MCAR.

---

**Null Hypothesis:** High-protein recipes (above median protein content) and low-protein recipes (below median protein content) have the same average rating, and any observed difference is due to random chance.

**Alternative Hypothesis:** High-protein recipes are, on average, rated 5 stars, compared to lower protein recipes

**Test Statistic:** Difference in proportion of 5 star ratings

**Significance Level:** p <= 0.05

**Results:**
- Observed difference: -0.036
- P-value: 1

**Conclusion:** We fail to reject the null hypothesis. The p-value of 1 is far above our significance level of 0.05. In fact, the observed difference is slightly negative, meaning high-protein recipes are not rated 5 stars, on average. There is no statistically significant evidence that higher-protein recipes receive 5 star ratings compared to lower-protein recipes.

---

# Framing a Prediction Problem

The prediction task is:

> **Can we predict whether a recipe will achieve a perfect five-star average rating?**

This is a **binary classification problem**.

### Response Variable

`five_star`

We chose this because understanding what makes a recipe five stars is directly useful for recommending and developing recipes people will enjoy.

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

The baseline model performs poorly. With an F1 of 0.53, the model only correctly predicts a 5 star recipe only half the time. Analyzing the quantitative features we had used on this model, we had only contributed measured features of the food recipe. We have still yet to touch on more abstract features like `tags`, preferential descriptive features like `ingredients`, time based measurements like `submitted_year`, all descriptions that can better capture patterns like general food.com user preferences, labels, membership between categories like "sweets", and larger societal trends towards food. All of these are avenues we plan to explore in the final model.

---

# Final Model

The final model extended the baseline using both engineered and textual features.

### Added Features

For the final model, we include all the numeric features, as well as features we engineered.

### `submitted_year`

During testing, we found that including datetime introduced too many different concentrations of data points, making prediction harder. We decided to simplify the feature further, engineering `submitted_year` to include a temporal aspect of the dataset. Since tastes and preferences change throughout the years, we were interested in if we could perform efficient classification for data found in previous years, that may or may not have an average rating assigned to it for some reason or another. This, we found, could simulate past trends in Food.com recipes. Secondly, we were interested in later analyzing fairness between older and newer recipes.

### `ingredient_text`
The original `ingredients` column contains the recipe's ingredients, contained in a list. In recent years, we observed outside of the dataset that due to the increase in short form content, 'trendy' recipes like Tiramisu, Dubai Chocolate Cookies, and others would have signature ingredients that are the main focus of the recipe. Inspired by this, we wanted to engineer the feature further, hypothesizing that some popular ingredients may improve model accuracy, both in past and present prediction. To accomplish this, we found that preprocessing the ingredients by replacing spaces with underscores, then combining the entire ingredient list into a string provides the equivalent of a document. This allowed us to perform TF-IDF on the dataset, defining each unique preprocessed recipe as a tag.

### `tag_text`
The `tag` column contains the recipe's user submitted tags, contained in a list. Tags functionally serve to distinguish a recipe by giving it short descriptors, aiding users and administrators in finding similar recipes and identifying a recipe's more abstract descriptors and features. Because of this, transforming the list into a string document and performing TF-IDF, we believed, would provide further refinement into giving the model insight into a recipe's abstracted features for classification purposes.

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

*Low-calorie recipes: calories at or below the median calorie count (305.4 calories)
High-calorie recipes: calories above the median calorie count

This grouping was chosen because calorie content is one of the primary nutritional features used by the model and may influence prediction behavior.

### Evaluation Metric

We compared the model's:

Precision
Recall
F1 Score

across both groups. Ideally, similar performance across groups would suggest that the model is not disproportionately favoring one calorie category over another.

### Null Hypothesis

The model is equally precise for low-calorie and high-calorie recipes. Any observed difference is due to random chance.

### Alternative Hypothesis

The model is less precise for high-calorie recipes.

### Method

A permutation test was performed using the difference in precision between groups as the test statistic.

### Conclusion

The results suggest that the classifier does not exhibit substantial bias toward either high-calorie or low-calorie recipes. While minor performance differences are expected due to natural variation in the data, there is no evidence that one calorie group is systematically disadvantaged by the model.

However, this analysis is limited to calorie-based groups. Future work could examine fairness across other recipe characteristics, such as preparation time, number of ingredients, cuisine tags, or dietary categories. Additionally, because recipe ratings reflect subjective user preferences, differences in group performance may arise from underlying rating patterns rather than model bias alone.
