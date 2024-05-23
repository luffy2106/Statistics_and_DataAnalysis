# Statistics_and_DataAnalysis

### 1. What is the A/B testing ?

In the context of machine learning, A/B testing refers to the process of comparing two machine learning models or algorithms to determine which one performs better in terms of a specific metric or key performance indicator (KPI). This approach is often used in real-world applications to decide which model to deploy or which algorithm to use for making predictions.

Here's how A/B testing works in the context of machine learning:
- Model Selection: Suppose you have two different machine learning algorithms (Model A and Model B) that you believe could be suitable for solving a particular problem. A/B testing helps you decide which model to choose for deployment.
- Data Splitting: You split your dataset into two parts: one part is used to train and evaluate Model A, and the other part is used to train and evaluate Model B. It's crucial to ensure that the data is split randomly and that both datasets are representative of the overall data distribution.
- Training and Evaluation: Model A and Model B are trained on their respective datasets. After training, both models are evaluated on the same evaluation metric (such as accuracy, precision, recall, etc.) using the same evaluation dataset.
- Comparison: The models' performance metrics are compared. Based on the comparison, you can determine which model performs better for the specific task.
- Decision Making: If one model significantly outperforms the other, the better-performing model is selected for deployment in the real-world application.
- Continuous Monitoring: A/B testing is not a one-time process. Models need to be continuously monitored, and if there are changes in the data distribution or if new algorithms are developed, A/B testing can be repeated to ensure that the deployed model remains the best choice.

A/B testing in machine learning is especially common in scenarios like algorithm selection, hyperparameter tuning, and feature engineering. It provides a data-driven approach to choosing the most effective solution for a given problem, ensuring that the chosen model is the one that optimizes the desired outcome.

Code Example:

```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming you have a dataset X (features) and y (target variable)

# Split the data into two random groups: A and B
X_A, X_B, y_A, y_B = train_test_split(X, y, test_size=0.5, random_state=42)

# Train Model A on Group A
model_A = RandomForestClassifier(random_state=42)
model_A.fit(X_A, y_A)

# Train Model B on Group B
model_B = RandomForestClassifier(random_state=42)
model_B.fit(X_B, y_B)

# Assuming you have a test set for evaluation
X_test, y_test = load_test_data()  # Function to load test data

# Evaluate Model A
predictions_A = model_A.predict(X_test)
accuracy_A = accuracy_score(y_test, predictions_A)
print("Accuracy of Model A:", accuracy_A)

# Evaluate Model B
predictions_B = model_B.predict(X_test)
accuracy_B = accuracy_score(y_test, predictions_B)
print("Accuracy of Model B:", accuracy_B)

# Compare the models and make a decision based on the accuracy scores
if accuracy_A > accuracy_B:
    print("Model A performs better. Deploy Model A.")
else:
    print("Model B performs better. Deploy Model B.")
```

#### 2.(Ekimetrics) How to calcualte overall ROI(Return for investment) ?

See the video interview for the question

ROI = (Current Value of Investment−Cost of Investment)/ Cost of Investment

For example, suppose Jo invested $1,000 in Slice Pizza Corp. in 2017 and sold the shares for a total of $1,200 one year later. 
To calculate the return on this investment, divide the net profits ($1,200 - $1,000 = $200) by the investment cost ($1,000), for an ROI of $200/$1,000, or 20%.

#### 3. What is the difference between dataframe and series
1. DataFrame:
- A DataFrame is a 2-dimensional labeled data structure, similar to a table or spreadsheet with rows and columns.
- It can store heterogeneous data types (e.g., integers, floats, strings) in its columns.
- Each column in a DataFrame is represented as a Series.
- You can think of a DataFrame as a collection of Series objects that share the same index.

2. Series:
- A Series is a 1-dimensional labeled array that can hold any data type, similar to a column in a spreadsheet or a single column of data in a DataFrame.
- It has an associated index, which labels the elements in the Series.
- A Series can be created from various data sources like lists, arrays, dictionaries, etc.
- Unlike a DataFrame, a Series does not have columns, it only has a single column of data.


#### [Sanofi] 4. What is the model coefficient in Machine Learning?

Model Coefficients in Machine Learning:
In machine learning, Model coefficients are values assigned to features by the algorithm to indicate their importance in making predictions. They represent the strength of the relationship between each feature and the target variable, helping interpret feature importance and model decisions.

For example, in linear regression, the model coefficients are the weights assigned to each input feature to determine the impact of each feature on the predicted output. 

The magnitude of the coefficient indicates the strength of the relationship between the feature and the target variable. A larger coefficient implies a stronger impact on the prediction.

#### [Sanofi] 5. What is Std Error in machine learning ?

Standard error in machine learning refers to the standard deviation of the sampling distribution of a statistic, such as coefficients or predictions. It measures the variability of estimates and helps assess the precision of the model parameters. Lower standard errors indicate more reliable and precise estimates.

Example :
Let's say you are fitting a linear regression model to predict house prices based on features like square footage, number of bedrooms, and location. After training the model, you obtain the following coefficients with their standard errors:
- Square Footage: 100 (Standard Error = 10)
- Number of Bedrooms: 20 (Standard Error = 5)
- Location: 30 (Standard Error = 15)

These standard errors indicate the variability in the estimates of the coefficients. For example, the coefficient for square footage has a lower standard error compared to the coefficient for location, suggesting that the estimate for square footage is more precise and reliable.

#### [Sanofi] 6. What is P-value in machine learning ?

In machine learning, the p-value is a statistical measure that helps determine the significance of a feature's contribution to the model. It indicates the probability of obtaining an effect (or result) as extreme as the one observed in the data under the assumption that the null hypothesis is true.
- If the p-value is low (typically less than 0.05), it suggests that the feature is statistically significant and likely has a meaningful impact on the target variable.
- If the p-value is high (greater than 0.05), it indicates that the feature may not be statistically significant and could potentially be removed from the model without affecting its performance significantly.

By analyzing p-values, machine learning practitioners can make informed decisions about feature selection and model interpretation based on the statistical significance of each feature.

Example :
Let's consider a scenario where we are building a machine learning model to predict housing prices based on various features such as square footage, number of bedrooms, and location.

After training the model, we calculate the p-values for each feature to determine their significance. Here are the results:

1. Square Footage:
- P-Value: 0.002
- Interpretation: The low p-value indicates that square footage is statistically significant in predicting housing prices.

2. Number of Bedrooms:
- P-Value: 0.123
- Interpretation: The relatively high p-value suggests that the number of bedrooms may not be statistically significant in predicting housing prices.
Location:

3. P-Value: 0.000
- Interpretation: The very low p-value indicates that location plays a significant role in predicting housing prices.
- Based on these p-values, we can prioritize features like square footage and location in our model, while potentially considering excluding the number of bedrooms if it does not significantly contribute to the predictive power of the model.

#### [Sanofi] 7. What is P-Intercept in the context of regression model (suppose that we have the example below)?
Ex :
A dataset contain sales of a music album in different media company : Spotify, Apple Music, and Youtube. You are provided marketing budget of album and have built a linear model to predict the sales . The model coefficients can bet found in the table below

|           | Coefficient | Std Error | P-Value  |
|-----------|-------------|-----------|----------|
| Intercept | 3.365       | 0.3119    | 0.000105 |
| Spotify   | 0.294       | 0.0075    | 0.000086 |
| Apple Music | 0.064     | 0.005     | 0.78     |
| Youtube   | 0.059       | 0.0011    | 0.00101  |

Ans :

In the context of a linear regression model, the Intercept term represents the value of the dependent variable (in this case, sales of the music album) when all independent variables (Spotify, Apple Music, Youtube) are set to zero.

- The Intercept coefficient in the table provided is 3.365.
- This means that if there is zero marketing budget allocated to Spotify, Apple Music, and Youtube, the expected sales value of the music album is 3.365 units.

Essentially, the Intercept accounts for the baseline or starting point of the dependent variable before considering the impact of the independent variables.

Therefore, the Intercept term provides the estimated value of the dependent variable when all independent variables are assumed to have no effect.


#### [Sanofi] 8. What is the difference between Coefficient vs P-Value in evaluating feature importance after training ML model?

Model coefficients indicate the strength and direction of the relationship between features and the target variable, while p-values help determine the statistical significance of these relationships. Both aspects are crucial in assessing feature importance in a machine learning model.

While a large coefficient suggests importance, the p-value helps confirm if this relationship is statistically significant. A feature with a high coefficient but a high p-value may not be reliable for prediction because [can skip if it's too long]:
-  Statistical Significance: The p-value provides a measure of how likely it is to observe the relationship between a feature and the target variable by random chance. A low p-value indicates that the relationship is unlikely to be due to randomness, making it more reliable for prediction.

- Accounting for Uncertainty: While a large coefficient may suggest importance, it doesn't guarantee that the relationship is significant. High coefficients can occur even with noisy or irrelevant features. The p-value helps account for this uncertainty by determining if the relationship is real or just a result of noise in the data.

- Avoiding Overfitting: A high coefficient without statistical significance could be a sign of overfitting, where the model is capturing noise in the training data rather than meaningful patterns. By considering the p-value, you can avoid over-relying on features that don't have a genuine impact on the target variable.

- Model Interpretation: A feature with a high coefficient but a high p-value may not provide actionable insights or contribute significantly to the model's predictive power. By focusing on features with both high coefficients and low p-values, you can prioritize those that are both influential and statistically significant.

- Generalization: Models built on features with low p-values are more likely to generalize well to unseen data. Features with high coefficients but high p-values may not generalize effectively, leading to poor performance on new data.


#### 9. What Is a T-Test in statistics ?

In statistics, a t-test is a type of inferential statistic used to determine if there is a significant difference between the means of two groups when the variances are unknown and sample sizes are small. It is commonly used when you have two groups and you want to determine if their means are truly different from each other or if those differences could have occurred by chance.

There are different types of t-tests such as:
1. Independent Samples T-Test: Used to compare the means of two independent groups.
Ex: 
One way to measure a person’s fitness is to measure their body fat percentage. Average body fat percentages vary by age, but according to some guidelines, the normal range for men is 15-20% body fat, and the normal range for women is 20-25% body fat.

2. Paired Samples T-Test: Used to compare the means of the same group at different times or under different conditions.
A paired samples t-test is commonly used in two scenarios:
- A measurement is taken on a subject before and after some treatment – e.g. the max vertical jump of college basketball players is measured before and after participating in a training program.
- A measurement is taken under two different conditions – e.g. the response time of a patient is measured on two different drugs.

3. One-Sample T-Test: Used to compare the mean of a sample to a known value or population mean.
Ex :  
- Null hypothesis : The average height of Vietnamese men is 1m70. 
- Alternative hypothesis(H1): The average height of Vietnamese men is 1m75

4. Equal Variance T-Test 
Equal Variance is conducted when the sample size in each group or population is the same, or the variance of the two data sets is similar.
Ex: 
- To compare the height of two male populations from the United States and Sweden, a sample of 30 males from each country is randomly selected and the measured heights are provided.

5. Unequal Variance T-Test  
Unequal Variance is used when the variance and the number of samples in each group are different. 
Ex : 
- To compare the height of two male populations from the United States and Sweden, a sample of 30 males from each country is randomly selected and the measured heights are provided.

Application:
- Widely applied in research studies, clinical trials, and experiments where sample sizes are limited or the population variance is unknown.

#### 10. What Is a Z-Test in statistics ?

In statistics, a Z-test is a hypothesis test that is used to determine whether the means of two populations are different when the variances are known and sample sizes are large. It is based on the standard normal distribution (Z-distribution) and is typically used for testing the mean of a single population or the difference between means of two populations.

Key Points:
- Population Variance Known: The Z-test is appropriate when the population variance is known.
- Large Sample Size: It is most reliable when the sample size is large (typically n > 30).
- Normal Distribution Assumption: The data should be approximately normally distributed.

Steps to Perform a Z-Test:
- Formulate Hypotheses: Define the null hypothesis (H0) and alternative hypothesis (H1).
- Calculate Test Statistic: Compute the Z-statistic using the formula appropriate for the type of Z-test being - conducted.
- Determine Critical Value: Compare the calculated Z-statistic with the critical value from the standard normal distribution based on the desired level of significance.
- Make a Decision: If the calculated Z-statistic falls within the rejection region, reject the null hypothesis; otherwise, do not reject it.

Types of Z-Tests:
- One-Sample Z-Test: Compares the mean of a single sample to a known population mean.
Ex : An online medicine shop claims that the mean delivery time for medicines is less than 120 minutes with a standard deviation of 30 minutes. Is there enough evidence to support this claim at a 0.05 significance level if 49 orders were examined with a mean of 100 minutes?

- Two-Sample Z-Test: Compares the means of two independent samples when the variances are known.
Ex : A teacher claims that the mean score of students in his class is greater than 82 with a standard deviation of 20. If a sample of 81 students was selected with a mean score of 90 then check if there is enough evidence to support this claim at a 0.05 significance level.


Applications of Z-Test:
- A/B Testing
- Quality Control
- Clinical Trials
- Market Research

Z-tests are powerful tools in statistical analysis when the conditions for their applicability are met, providing valuable insights into population parameters based on sample data.

#### 11. What is the difference between T-test and Z-test ?
1. Based on Sample Size and Population Variance:
T-test:
- Does not require knowledge of the population variance.
- Suitable for smaller sample sizes (n < 30) or when the population variance is unknown.
Z-test:
- Requires that the population variance is known.
- Typically used when the sample size is large (n > 30).
2. Type of Distribution
T-test:
- Utilizes the Student's t-distribution.
- Accounts for the additional uncertainty introduced by estimating the population standard deviation from the sample.
Z-test:
- Relies on the standard normal distribution (Z-distribution).
- Appropriate when data is normally distributed.
3. Application
T-test:
- Widely applied in research studies, clinical trials, and experiments where sample sizes are limited or the population variance is unknown.
Z-test:
- Commonly used in situations where the population variance is known and sample size is large, such as quality control and A/B testing.

#### 12. what is Pearson Correlation Coefficient in statistics ?

The Pearson correlation coefficient, also known as Pearson's r, is a measure of the correlation coefficient and linear relationship between two "continuous variables". It ranges from -1 to 1, where:
- 1 indicates a perfect positive linear relationship
- -1 indicates a perfect negative linear relationship
- 0 indicates no linear relationship

The formula for calculating the Pearson correlation coefficient (r) is:

r = (Σ((x - mean(x)) * (y - mean(y)))) / (sqrt(Σ(x - mean(x))^2) * sqrt(Σ(y - mean(y))^2))

In this formula:
- Σ denotes the sum of the values
- x and y are the two variables being compared
- mean(x) and mean(y) are the means (averages) of x and y, respectively
- sqrt denotes the square root

#### 13. What Is a Kendall Correlation Coefficient(Kendall's Tau) in statistics ?
The Kendall correlation coefficient, also known as Kendall's tau  is a measure of association between two measured quantities. It assesses the ordinal association between two variables without assuming any linearity of relationships.

Kendall's tau can be used when data are "ranked" rather than measured on a continuous scale. It ranges from -1 to 1, where:
- 1 indicates a perfect agreement in rankings,
- -1 indicates a perfect disagreement in rankings,
- 0 indicates no association between the rankings.

Ex : 
Observation X Rank Y Rank
E           3      4     
F           1      3     
G           2      1     
H           4      2     

Using the formula for Kendall's tau:
tau = (number of concordant pairs - number of discordant pairs) / (n(n-1) / 2)

In this case:
- tau = (3 - 3) / (4(4-1) / 2)
- tau = 0 / 6
- tau = 0

So, in this example, Kendall's tau coefficient between variables X and Y is 0, indicating no association between the rankings of X and Y.

#### 14. what is Chi-Square Test in statistics ?

The Chi-Square Test is a statistical test that is used to determine whether there is a significant association between "two categorical variables". It is commonly used to analyze data that consists of counts or frequencies for different categories.
Key Points:
- The null hypothesis of the Chi-Square Test states that there is no significant association between the variables.
- The test compares the observed frequencies in the data with the frequencies that would be expected if the variables were independent.
- The test statistic is calculated as the sum of the squared differences between the observed and expected frequencies, divided by the expected frequencies.
- The p-value associated with the test statistic is used to determine the significance of the results. A low p-value indicates that the variables are likely to be associated.

Chi-Square Test is widely used in various fields such as biology, social sciences, market research, and quality control to analyze categorical data and determine if there is a significant relationship between variables.

Ex :
When we want to determine if there is a significant association between gender and smoking status among a group of individuals.
Hypothesis:
- Null Hypothesis (H0): There is no significant association between gender and smoking status.
- Alternative Hypothesis (H1): There is a significant association between gender and smoking status.

#### 15. what is Cramér's V in statistics ?

Cramér's V is a measure of association between two categorical variables. It is based on the chi-squared statistic and varies between 0 and 1, where:
- 0 indicates no association between the variables
- 1 indicates a perfect association between the variables



#### 16.(Ekimetrics) Which technique you use to evaluate correlation between two categorial variables ?
1. Calculate the pearson correlation coefficient 
2. Apply a T-test
3. Calculate the kendall correlation coefficient 
4. Apply a Chi-2 test

Recall : 
- pearson coefficient : A statistics measurement to evaluate correlation between two "continuous variable"(see more on question 12)
- Kendall Correlation Coefficient(Kendall's Tau) :  A statistics measurement to evaluate the relationship between two columns of "ranked data".(See question 13)
- T-test : a type of inferential statistic used to determine if there is a significant difference between the "means" of two groups. It is commonly used when you have two groups and you want to determine if their means are truly different from each other or if those differences could have occurred by chance. (See question 9)
- Chi-2 test : a statistical test that is used to determine whether there is a significant association between two "categorical variables". (See question 14)

=> To evaluate correlation between two categorial variable, we choose "Chi-2 test" method.

Note :
- Pearson coefficient, kendall and Cramér's V use threshold to measure relationship between 2 variables(Pearson coefficient is for continuous variable, kendall is for ranking variable, and Cramér's V is for categorial variable )
- t-test and chi 2 test use hypothesis testing to see if there is association between 2 varialbes(there will be null hypothesis and alternative hypothesis). But t-test is for continous variable and chi 2 test is for categorial variable. 


#### 17. What is the common and difference between T-test, Z-test, P-value method and Chi2 test ?
a. Common : 
- T-test, Z-test, P-value method and Chi2Test used for hypothesis testing in statistics
b. Difference :
- T-test, Z-test, P-value method used for continuous variable.
- Chi2Test method used for categorial variable.


#### 18. Give me some probability distribution you know?
- Bernoulli Distribution: A Bernoulli distribution has only two possible outcomes, namely 1 (success) and 0 (failure), and a single trial. So the random variable X which has a Bernoulli distribution can take value 1 with the probability of success, say p, and the value 0 with the probability of failure, say q or 1-p. The probability mass function is given by: p^x(1-p)^1-x  where x € (0, 1). 
Example: when you flip a coin.

- Uniform Distribution: The probabilities of getting these outcomes are equally likely and that is the basis of a uniform distribution. Unlike the Bernoulli Distribution, all the n number of possible outcomes of a uniform distribution are equally likely.
Example : When you roll a fair die, the outcomes are 1 to 6.

- Binomial Distribution: come back to the coin flip, but this time you conduct many toss (Bernoulli Distribution is a special case of Binomial Distribution with a single trial). Each trial is independent since the outcome of the previous toss doesn’t determine or affect the outcome of the current toss. An experiment with only two possible outcomes repeated n number of times is called binomial. The parameters of a binomial distribution are n and p where n is the total number of trials and p is the probability of success in each trial. 
Example:
Suppose we have a biased coin that comes up heads with a probability of 0.6 (success) and tails with a probability of 0.4 (failure). We want to determine the probability of getting exactly 3 heads in 5 tosses of this coin.
Calculation:
* Number of trials (n): 5 (tossing the coin 5 times)
* Probability of success (p): 0.6 (getting heads)
* Number of successes (k): 3 (getting exactly 3 heads)

- Normal distribution(Gaussian): has a bell shape, which is used to see how much data observed are different from the mean.

- Poisson Distribution: In case you want to do statistics about how many calls you get in a day. A distribution is called a Poisson distribution when the following assumptions are valid:
1. Any successful event should not influence the outcome of another successful event.
2. The probability of success over a short interval must equal the probability of success over a longer interval.
3. The probability of success in an interval approaches zero as the interval becomes smaller.

- Exponential Distribution: Let’s consider the call center example one more time. What about the interval of time between the calls? Here, the exponential distribution comes to our rescue. Exponential distribution models the interval of time between the calls.
Other examples are:
1. Length of time between metro arrivals,
2. Length of time between arrivals at a gas station
3. The life of an Air Conditioner

The exponential distribution is widely used for survival analysis. From the expected life of a machine to the expected life of a human, exponential distribution successfully delivers the result.

```
https://www.analyticsvidhya.com/blog/2017/09/6-probability-distributions-data-science/
```

#### 19. What is an estimate and estimator in statistics?
- An estimator is the method or procedure used to calculate an estimate, 
- while an estimate is the specific value obtained by applying the estimator to the sample data. 

Estimators help us infer information about populations based on sample data by providing estimates of unknown parameters.

#### 20. What is the basic statistics estimator you know?
- Mean
- Median
- Mode
- standard deviation
- z-score and p-value

#### 21. What are the properties of statistics estimator you know?
- Bias: Bias refers to the expected difference between the estimated value of the parameter (on a specific sample) and the “true” one (in the true model). 
- Variance :  Variance tells us how much data spread from the mean
- Efficiency: Efficiency means, that if the sample size increases, then the estimated parameters will not change substantially, they will vary in a narrow range (variance of estimates will be small). In the case of inefficient estimates, the increase of sample size from 50 to 51 observations may lead to the change of a parameter from 0.1 to, let’s say, 10. 
- Consistency: Consistency means that our estimates of parameters will get closer to the stable values (true value in the population) with the increase of the sample size. 

#### 22. How we can describe the characteristics of a distribution, especially in normal distribution?
In Statistics, Moments are popularly referring to quantities that describe the characteristics of a distribution. These are very useful in statistics because they tell you much about your data.
The four commonly used Moments in statistics are:
- the mean(The First Moment): The first central moment is the expected value, known also as an expectation, mathematical expectation, mean, or average.
- variance(Second moment): Variance represents how a set of data points are spread out around their mean value.
- skewness(Third moment): It measures how asymmetric the distribution is about its mean. three types of distribution with respect to its skewness:
* Symmetrical distribution: If both tails of a distribution are symmetrical, and the skewness is equal to zero, then that distribution is symmetrical.
* Positively Skewed: the right tail (with larger values) is longer. So, this also tells us about ‘outliers’ that have values higher than the mean
* Negatively skewed: the left tail (with small values) is longer. So, this also tells us about ‘outliers’ that have values lower than the mean
- kurtosis(Fourth moment): It measures the amount in the tails and outliers. It focuses on the tails of the distribution and explains whether the distribution is flat or rather with a high peak. 

#### 23. What is the property of normal distribution/gaussian distribution?
- It's symmetric and has the shape of a bell.
- The total area under the curve is 1
- Empirical rule: 
In normally distributed data, there is a constant proportion of distance lying under the curve between the mean and a specific number of standard deviations from the mean. For example:
* 68.25% of all cases fall within +/- one standard deviation from the mean.  μ - σ to μ + σ 
* 95% of all cases fall within +/- two standard deviations from the mean. μ - 2σ to μ + 2σ 
* 99% of all cases fall within +/- three standard deviations from the mean. μ - 3σ to μ + 3σ
- Skewness and kurtosis
* Skewness measures the symmetry of a normal distribution.
* kurtosis measures the thickness of the tail ends relative to the tails of a normal distribution.
- If we take simple random samples (with replacement)of size from the data that follow a normal distribution and compute the mean for each of the samples, the distribution of sample means should be approximately normal distribution according to the Central Limit Theorem. See the details here:
```
https://www.scribbr.com/statistics/central-limit-theorem/
```

Reference:
```
https://corporatefinanceinstitute.com/resources/knowledge/other/normal-distribution/
```

#### 24. What is the Central Limit Theorem?
The Central Limit Theorem (CLT) is a fundamental concept in statistics that states the following:
- Regardless of the shape of the population distribution, the sampling distribution of the sample mean approaches a normal distribution as the sample size gets larger.

Key Points:
- The Central Limit Theorem applies to the distribution of sample means or sums, not the original population.
- It is essential for inferential statistics because it allows us to make inferences about population parameters using sample data.
- The CLT holds true even if the population distribution is not normal, as long as the sample size is sufficiently large.
- As the sample size increases, the sampling distribution of the sample mean becomes increasingly normal, with the mean of the sample means equal to the population mean.

Implications:
- It enables us to use the normal distribution and related statistical techniques for hypothesis testing, confidence intervals, and making population inferences.
- It provides a justification for why many statistical methods rely on the assumption of normality, even when the underlying population distribution is not normal.

In conclusion, the Central Limit Theorem is a crucial concept in statistics that explains how the distribution of sample means converges to a normal distribution as the sample size grows, allowing for reliable statistical inference even when the population distribution is unknown or non-normal.

For illustration, take a look at this picture:
```
https://www.scribbr.com/statistics/central-limit-theorem/
```

#### 25. Explain hypothesis testing 
[Check goodnote for more details]


[Check goodnote for more questions]

