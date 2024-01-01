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


#### (Ekimetrics) How to calcualte overall ROI(Return for investment) ?

See the video interview for the question

ROI = (Current Value of Investment−Cost of Investment)/ Cost of Investment

For example, suppose Jo invested $1,000 in Slice Pizza Corp. in 2017 and sold the shares for a total of $1,200 one year later. 
To calculate the return on this investment, divide the net profits ($1,200 - $1,000 = $200) by the investment cost ($1,000), for an ROI of $200/$1,000, or 20%.

#### (Ekimetrics) Which technique you use to evaluate correlation between two categorial variables ?
1. Calculate the pearson correlation coefficient 
2. Apply a T-test
3. Calculate teh kendall correlation coefficient 
4. Apply a Chi-2 test
(at the beginning I choose 1, it's seem to be wrong)

To evaluate correlation between two continuous variable, we can use:
- pearson coefficient : the formular based on covariance and standard deviation.
  * 1 indicates a strong positive relationship.
  * -1 indicates a strong negative relationship.
  * 0 indicates no relationship at all
- kendall correlation : best use for ranking features 
- t-test :  the formular based on observed sample mean, the theoretical population means, sample standard deviation, and sample size. 

To evaluate correlation between two categorial variable, we can use :
- chi-2 test :  Test of Significance to determine the difference observed and expected frequencies of certain observations
- Cramér's V

Note :
- Pearson coefficient, kendall and Cramér's V use threshold to measure relationship between 2 variables.
- t-test and chi 2 test use hypothesis testing to see if there is association between 2 varialbes(there will be null hypothesis and alternative hypothesis). But 
t-test is for continous variable and chi 2 test is for categorial variable. 


reference :
- https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/
- https://www.statisticshowto.com/kendalls-tau/
- https://www.educba.com/t-test-formula/
- https://sites.utexas.edu/sos/guided/inferential/categorical/chi2/
- https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=terms-cramrs-v


#### What is the common and difference between T-test, Z-test, P-value method and Chi2 test ?
##### Common : 
T-test, Z-test, P-value method and Chi2Test used for hypothesis testing in statistics
##### Difference :
- T-test, Z-test, P-value method used for continuous variable.
- Chi2Test method used for categorial variable.
###### Z-test : 
A z test is a statistical test that is used to check if the means of two data sets(Two Sample Z Test), or the sample mean and the population mean(One-Sample Z Test), are different when the population variance is known. This test used to dealing with problems relating to large sample(n>30).
- Two Sample Z Test : to check the difference of 2 datasets. Ex : A teacher claims that the mean score of students in his class is greater than 82 with a standard deviation of 20. If a sample of 81 students was selected with a mean score of 90 then check if there is enough evidence to support this claim at a 0.05 significance level.
- One-Sample Z Test : An online medicine shop claims that the mean delivery time for medicines is less than 120 minutes with a standard deviation of 30 minutes. Is there enough evidence to support this claim at a 0.05 significance level if 49 orders were examined with a mean of 100 minutes?

How to perform Z-test(use Z-table) <br />
https://www.cuemath.com/data/z-test/

###### T-test :
- A t-test is used to check if the means of two data sets are different when the population variance is not known.This test is used to dealing with problems relating to small sample(n<30). Calculating a t-test requires three fundamental data values including the difference between the mean values from each data set, the standard deviation of each group, and the number of data values.

5 types of T-Test
1. One-Sample T-Test : One-sample is used to determine whether an unknown population mean is different from a specific value.
Ex: Null hypothesis : The average height of Vietnamese men is 1m70. Alternative hypothesis(H1): The average height of Vietnamese men is 1m75

2. Independent Two-Sample T-Test : An independent Two-Sample test is conducted when samples from two different groups, species, or populations are studied and compared.
Ex: One way to measure a person’s fitness is to measure their body fat percentage. Average body fat percentages vary by age, but according to some guidelines, the normal range for men is 15-20% body fat, and the normal range for women is 20-25% body fat.

3. Paired Sample T-Test: Paired Sample is the hypothesis testing conducted when two groups belong to the same population or group.
A paired samples t-test is commonly used in two scenarios:

- A measurement is taken on a subject before and after some treatment – e.g. the max vertical jump of college basketball players is measured before and after participating in a training program. 
- A measurement is taken under two different conditions – e.g. the response time of a patient is measured on two different drugs.
4. Equal Variance T-Test : Equal Variance is conducted when the sample size in each group or population is the same, or the variance of the two data sets is similar.
Two-sample T-Test with equal variance can be applied when

- the samples are normally distributed,
- the standard deviation of both populations are unknown and assumed to be equal, and
- the sample is sufficiently large (over 30).
Ex : To compare the height of two male populations from the United States and Sweden, a sample of 30 males from each country is randomly selected and the measured heights are provided.

5. Unequal Variance T-Test : Unequal Variance is used when the variance and the number of samples in each group are different.
Two-sample T-Test with unequal variance can be applied when
- the samples are normally distributed,
- the standard deviation of both populations are unknown and assume to be unequal,
- sample is sufficiently large (over 30).
Ex : To compare the height of two male populations from the United States and Sweden, a sample of 30 males from each country is randomly selected and the measured heights are provided.

How to perform T-test(use T-table) <br />
https://www.educba.com/t-test-formula/

###### P-value method :  
While the T values and Z values indicate the chances of the difference between the sample means. P-value is the assumption test used to negate the fact that the means of two samples have no difference.

How to perform P-value method(use Z-table)  <br />
https://towardsdatascience.com/p-value-method-for-hypothesis-testing-c967c0f78d74


###### Chi2-test :
Test of Significance to determine the difference observed and expected frequencies of certain observations. This method is used to compare the correlation 
between 2 categorial variables.

How to performe ChiSquare(use Chi2 table) <br />
https://www.simplilearn.com/tutorials/statistics-tutorial/chi-square-test


###### Example about Pearson correlation

The Pearson correlation formula is used to calculate the correlation coefficient between two variables. The correlation coefficient measures the strength and direction of the linear relationship between two variables.

The formula for calculating the Pearson correlation coefficient (r) is:

r = (Σ((x - mean(x)) * (y - mean(y)))) / (sqrt(Σ(x - mean(x))^2) * sqrt(Σ(y - mean(y))^2))

In this formula:
- Σ denotes the sum of the values
- x and y are the two variables being compared
- mean(x) and mean(y) are the means (averages) of x and y, respectively
- sqrt denotes the square root

Exercice example(Aily lab test):
```
2 bateria cutures, A and B,were set up in two different dishes, each covering 50% of its dish. Over 20 days, bateria A's percentage of coverage increased to 70% and bateria B's percentage of coverage reduced to 40%.
Approximately, what is the Pearson correlation coefficient of bateria B's coverage and the number of days passed ?
```

Solution:
```
To calculate the Pearson correlation coefficient between Bacteria B's coverage and the number of days passed, we need to determine the relationship between these two variables.

In this case, we can observe that as the number of days passed increases, Bacteria B's coverage reduces. This implies a negative correlation between the two variables.

To approximate the Pearson correlation coefficient, we can use the given information:

Bacteria B's initial coverage is 50%.
Bacteria B's final coverage is 40%.
The number of days passed is 20.
Using the formula mentioned earlier:

r = (Σ((x - mean(x)) * (y - mean(y)))) / (sqrt(Σ(x - mean(x))^2) * sqrt(Σ(y - mean(y))^2))

We can substitute the values:

x = [initial coverage, final coverage] = [50%, 40%]
y = [number of days passed] = [0, 20]

mean(x) = (50% + 40%) / 2 = 45%
mean(y) = (0 + 20) / 2 = 10

Σ((x - mean(x)) * (y - mean(y))) = ((50% - 45%) * (0 - 10)) + ((40% - 45%) * (20 - 10)) = (-5% * -10) + (-5% * 10) = 0

Σ(x - mean(x))^2 = ((50% - 45%)^2) + ((40% - 45%)^2) = (5%)^2 + (5%)^2 = 0.05% + 0.05% = 0.1%

Σ(y - mean(y))^2 = ((0 - 10)^2) + ((20 - 10)^2) = (10)^2 + (10)^2 = 100 + 100 = 200

Now, we can substitute these values into the formula:

r = (0) / (sqrt(0.1%) * sqrt(200))
r = 0 / (0.316% * 14.142)
r = 0 / 0.447
r ≈ 0

Therefore, the approximate Pearson correlation coefficient of Bacteria B's coverage and the number of days passed is 0, indicating no correlation between the two variables.
```

###### What is the difference between dataframe and series
1. DataFrame:
A DataFrame is a 2-dimensional labeled data structure, similar to a table or spreadsheet with rows and columns.
It can store heterogeneous data types (e.g., integers, floats, strings) in its columns.
Each column in a DataFrame is represented as a Series.
You can think of a DataFrame as a collection of Series objects that share the same index.

2. Series:
A Series is a 1-dimensional labeled array that can hold any data type, similar to a column in a spreadsheet or a single column of data in a DataFrame.
It has an associated index, which labels the elements in the Series.
A Series can be created from various data sources like lists, arrays, dictionaries, etc.
Unlike a DataFrame, a Series does not have columns; it only has a single column of data.
