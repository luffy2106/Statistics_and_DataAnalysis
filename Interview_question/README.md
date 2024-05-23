# Interview question


[Sanofi] Question :
A dataset contain sales of a music album in different media company : Spotify, Apple Music, and Youtube. You are provided marketing budget of album and have built a linear model to predict the sales . The model coefficients can bet found in the table below :

|           | Coefficient | Std Error | P-Value  |
|-----------|-------------|-----------|----------|
| Intercept | 3.365       | 0.3119    | 0.000105 |
| Spotify   | 0.294       | 0.0075    | 0.000086 |
| Apple Music | 0.064     | 0.005     | 0.78     |
| Youtube   | 0.059       | 0.0011    | 0.00101  |


Which independent variable is least significant ?
- Youtube
- Sales
- Spotify
- Apple music

Solution :
Recall that P-value is statistical measure that helps determine the significance of a feature's contribution to the model. The smaller the p-value, the more important the feature is. Also remember that while Coefficient can determine the relationship between variables and target but it's not the prority compare to P-value(see README.md for more details)

The independent variable with the highest p-value (and thus the least significance) is Apple Music with a p-value of 0.78. This suggests that the marketing budget spent on Apple Music promotion may not have a statistically significant impact on the sales of the music album compared to Spotify and Youtube.


[Sanofi] Question :
A study was done in a lab on the O2 levels in the blood of man versus women, it was found that 4% of the women had over 85 mmHg O2 in their bloodstream and 1% of men had over 85 mmHg O2 in their bloodstream. The total population in the lab was 60% in favor of men. If one individuals selectd randomly among those who had over 85 mmHg O2 in their bloodstream, what is the probability that this individual is a man ?

Solution :
The problem can be rephrase as :
P(Y=has O2 > 85mmHg/X = woman) = 0.04 
P(Y=has O2 > 85mmHg/X = man) = 0.01
P(X = man) = 0.6
P(X = woman) = 0.4

P(X=man/Y=has O2 > 85mmHg) = ?

Start calculating :

Apply Bayes's Theory, we find:

P(X=man/Y=has O2 > 85mmHg) = P(X=man and Y=has O2 > 85mmHg) / P(Y=has O2 > 85mmHg)

While :
- P(X=man and Y=has O2 > 85mmHg) = P(Y=has O2 > 85mmHg/X = man) * P(X = man) = 0.01 * 0.6 = 0.006
- P(Y=has O2 > 85mmHg) = P(Y=has O2 > 85mmHg/X = man) * P(X = man) + P(Y=has O2 > 85mmHg/X = woman) * P(X = woman) = 0.01 * 0.6 + 0.04 * 0.4 = 0.022

=> P(X=man/Y=has O2 > 85mmHg) = 0.006 / 0.022 = 27.27%

[Sanofi] Question : 
what should we do with loss function if class is imbalanced ?

Solution:
Here are some strategies for handling imbalanced classes in the loss function:
1. Class Weights:
Assign higher weights to the samples from the minority class in the loss function. This way, the model pays more attention to correctly classifying the minority class.
2. Balanced Loss Functions:
Use loss functions specifically designed to handle imbalanced classes, such as weighted cross-entropy loss or focal loss. These loss functions automatically adjust the contribution of each class based on their frequencies.
3. Sampling Techniques:
Utilize techniques like oversampling (replicating samples from the minority class), undersampling (removing samples from the majority class), or more advanced methods like SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution.
4. Ensemble Methods:
Combine multiple models trained on balanced subsets of the data to create an ensemble model that can better generalize across imbalanced classes.
5. Evaluation Metrics:

Look beyond accuracy and consider using evaluation metrics like precision, recall, F1-score, or area under the ROC curve (AUC) that are more informative for imbalanced datasets.

[Sanofi] Question :

In performing a 5 fold cross validation on a machin learning that has 2 parameters, we want to test 3 different values for the former hypeparameter and 2 different values for latter. How many models need to be trained to identify the best hyperparameter set ?

Solution:

1. For each combination of hyperparameters, you will need to train:
- 5 models (one for each fold in the cross-validation)
2. Calculate the total number of combinations of hyperparameters:
- Total combinations = Number of values for hyperparameter 1 * Number of values for hyperparameter 2
                     = 3 * 2
                     = 6
3. Multiply the total combinations by the number of models per combination:
- Total models = Total combinations * Number of models per combination
               = 6 * 5
               = 30

Therefore, you will need to train 30 models in total to identify the best hyperparameter set using 5-fold cross-validation with 3 different values for one hyperparameter and 2 different values for the other hyperparameter.

[Sanofi] Question:

which of the listed properties are generally considered good for cost function of the models ?

- When the function is odd
- WHen the function is convex
- When the function is cheap to calculate
- When the functiion is differentiable
- None of the above

Select all correct options

Note(*): 
In the context of mathematical functions, a function can be classified as odd if it satisfies the property:

[f(-x) = -f(x)]

Solution
Out of the listed properties, the following are generally considered good for the cost function of models:
1. When the function is convex
A convex cost function is desirable because it guarantees a single global minimum, making optimization more straightforward and less prone to getting stuck in local minima.
2. When the function is differentiable
Differentiability is essential for gradient-based optimization algorithms like gradient descent to converge efficiently towards the optimal solution.
3. When the function is cheap to calculate
A function is cheap to calculate will reduce the time complexity and space complexity of the model

[Sanofi] Question

You can generate hundered of individual features trees in a Random Forest, Choose the correct statement about an individual tree in Random Forest

1. An individual tree is built on a subset of the features
2. An individual tree is built on all the features
3. An individual tree is built on a subset of observations
4. An individual tree is built on the full set of obeservation

Solution:

Remember that the randomness in random forest is on
- The subset of dataset : Each individual tree in a Random Forest is trained on a random subset of the training data (bootstrap sample) to introduce diversity and reduce overfitting.
- the subset of the features : Each individual tree in a Random Forest is trained on a random subset of features as well

overfitting.

so (1) and (2) is correct.





