# Food.com Five Star Recipe Analysis:

**Author:** Adrianne See, Aarman Sachdev

---
# Overview

This Data Science project explores the effectiveness of predictive models in understanding a 'Five Star Rating' for any given user recipe from Food.com.

# Introduction

Food.com is one of the largest food preparation websites on the Internet, hosting over 500,000 user made recipes from around the world, as well as feedback, reviews and images of other users making them at home. Launching in 2017 to great success, its recipes have since garnered an active community of home cooks looking to enjoy a good recipe. Food.com is not home to just home users, though, as many celebrities, chefs, and popular companies have both derived and official recipes published as well. 

Because of this, understanding what drives a 'Five Star Rating' becomes highly important. From promotional avenues, to publicity stunts, to establishing a mini community, individuals and businessses can serve to profit from releasing well received recipes on this website. Numerical insights can also help home cooks and food platforms recommend meals that people will actually enjoy. Depending on factors such as nutrition content, calorie density, and simplicity of preparation, general trends in food preferences can also be discovered. Finally, trends towards ingredients or tags can help any recipe maker promote or create recipes that are timely and well received by users.

This project investigates the following question:

> **What characteristics are associated with five-star recipes on Food.com, and can recipe metadata be used to predict whether a recipe will receive a perfect rating?**

To perform this analysis, this project analyzes a subset of  user recipes and  user ratings from Food.com. Each recipe includes nutritional information, preparation details, and user-submitted ratings. 

To perform this analysis, this project analyzes a subset of  user recipes and  user ratings from Food.com. Each recipe includes nutritional information, preparation details, and user-submitted ratings. 

The first dataset we want to leverage, `recipe`, contains a subset (83,782) of all user submitted recipes uploaded to Food.com. Each entry represents a unique recipe submitted and viewable on the website, and contains the following features:

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

The second dataset we want to leverage, `ratings`, contains a subset (731,927) of all user submitted ratings uploaded to Food.com. Each entry represents a unique review for a unique recipe on the website, and contains the following features:

| Column Name | Description | 
|--------|-------------------|
| `user_id'` | User ID |
| `recipe_id` | Recipe ID |
| `date` | Date of interaction |
| `rating`	 | Rating given |
| `review` | Review text |

**Relevant Columns:**
- `calories`: extracted from the `nutrition` column, represents total calories per serving
- `avg_rating`: average user rating for each recipe (1-5 scale)
- `n_steps`: number of steps in the recipe
- `n_ingredients`: number of ingredients
- `minutes`: total preparation time
- `tags`: user assigned tags to describe recipe
- `ingredients`: Ingredients
---

# Data Cleaning

While useful, the dataset is split into multiple parts. To simplify analysis and prediction, we will combine the information into one usable and analyzable dataset. To do so, teh following transformations were done.

1. **Merged recipes and interactions:** We performed a left merge on the `recipes` dataset with the `interactions` dataset, on `id` for recipes and `recipe_id` with interactions. This was done to associate each recipe with a rating column.

2. **Replaced 0 ratings with NaN:** Ratings of 0 were replaced with `np.nan`. After testing, a user account was unable to give a rating of 0 stars. This supports the conclusion that recipe ratings of zero stars are more likely due to data error or lack of user rating, rather than genuine ratings. Replacing 0's with `np.nan` prevents them from skewing the rating values when computing the mean rating per recipe.

3. **Computed average rating per recipe:** The mean rating for each recipe was calculated and assigned onto the `recipes` dataset as `avg_rating`.

4. **Parsed nutrition column:** The `nutrition` column stores `calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV)`, with the data looking like a list stored in string format. We converted these to actual lists and extracted individual nutritional values, creating the appropriate column names.

5. **Decimal transform of non caloric nutrition columns:** `total fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, and `carbohydrates` are all in Percentage Daily Value (PDV). For our analysis purposes, we found it useful to convert the percentages to decimals.

6. **Parsed tags column:** The `tags` column contains tags for the recipe, contained as a list in string format. These were converted into lists for future use. 

7. **Extracted Recipe Creation Year into `submitted_year`:** The year of recipe submission is present in the Dataset, but contaiend wihtin the submission date. This feature was extracted and put into `submitted_year` for better analysis and feature use.

8. **Create Target Column, ``five_star``:** The target prediction is the group of ratings that are equal to 5, or 5.0. A simple boolean mask was performed on `avg_rating` to produce the target column.

The resulting main DataFrame, Recipes, now has 83782 entries and 22 features. Below is a table of all the columns

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

From here, special attention will be given to  the following columns

- `minutes`
- `tags`
- `nutriton`, and its resulting columns `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrates`
- `submitted_year`

## Univariate Analysis

<iframe
 src="assets/uni-calories.html"
 width="100%"
 height="600"
 frameborder="0">
</iframe>

For this analysis, we wanted to carefully analyze the distribution of caloric count in a recipe. The histogram below shows the distribution of calories across all recipes from our sample of Food.com. The good amount of the recipes (75.1%), contain strictly less than 500 calories, and a majority (94.2%) contain strictly less than 1000 calories. The graph is also right skewed, indicating that increasingly fewer recipes on Food.com have higher caloric count. We found it also useful to call the describe function on the 'calories' column, whose results are stored below. Of note is the effect of the skew, with the mean (429.92 calories) sitting above the median (305.4 calories), reinforcing the conclusion of the skewed nature of the dataset.

## Bivariate Analysis

<iframe
 src="assets/bivariate-rating-calorie.html"
 width="100%"
 height="600"
 frameborder="0">
</iframe>

For our Bivariate Analysis, we wanted to compare the ratings of lower and higher caloric recipes. The boxplot compares recipe ratings across low-calorie and high-calorie recipes using the median calorie count as a threshold. Both groups exhibit very similar rating distributions. Calorie content alone appears to have limited predictive power for determining whether a recipe receives a perfect rating.

## Interesting Aggregates

The table shows the distribution of number of ingredients per recipe, as well as the resulting average calories, sugar, and average rating for each ingredient count. Recipes with only 1 ingredient tend to have much higher calories (714.65) and higher ratings (4.86) on average. As the number of ingredients increases, both calories and ratings slightly decrease. Interestingly, recipes with very few ingredients tend to receive slightly higher ratings. This may reflect the pattern that recipes with less ingredients may be both more calorie-dense and reviewed more ratings. Sugar follows a roughly decreasing trend, while protein seems to increase as ingredients increase.

| n_ingredients | calories | total_fat | sugar | sodium | protein | avg_rating |
|--------------:|---------:|----------:|------:|-------:|--------:|-----------:|
| 1  | 714.65 | 0.91 | 1.04 | 0.26 | 0.25 | 4.86 |
| 2  | 331.98 | 0.26 | 0.80 | 0.53 | 0.18 | 4.69 |
| 3  | 306.11 | 0.22 | 0.78 | 0.23 | 0.16 | 4.66 |
| 4  | 336.01 | 0.24 | 0.87 | 0.18 | 0.20 | 4.63 |
| 5  | 347.80 | 0.25 | 0.78 | 0.25 | 0.22 | 4.65 |
| 6  | 355.89 | 0.26 | 0.71 | 0.23 | 0.24 | 4.63 |
| 7  | 391.13 | 0.30 | 0.64 | 0.28 | 0.29 | 4.62 |
| 8  | 387.03 | 0.30 | 0.55 | 0.26 | 0.31 | 4.61 |
| 9  | 422.50 | 0.32 | 0.62 | 0.27 | 0.33 | 4.61 |
| 10 | 449.82 | 0.35 | 0.63 | 0.29 | 0.36 | 4.61 |

---

# Assessment of Missingness

## NMAR Analysis

The `description` column contains 70 missing values.

e believe this is likely NMAR, as recipe contributors who put less effort into their submission may skip writing a description, and those same contributors may also submit lower quality recipes. The missingness is therefore related to the actual value of the description itself (its quality/existence), not just other observed columns.

Additional information such as contributor account age or posting history could potentially explain this missingness mechanism.

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
