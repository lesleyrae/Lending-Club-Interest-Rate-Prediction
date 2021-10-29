# Lending-Club-Interest-Rate-Prediction

- Description
 - This data set represents thousands of loans made through the Lending Club platform, which is a platform that allows individuals to lend to other individuals. Of course, not all loans are created equal. Someone who is a essentially a sure bet to pay back a loan will have an easier time getting a loan with a low interest rate than someone who appears to be riskier. And for people who are very risky? They may not even get a loan offer, or they may not have accepted the loan offer due to a high interest rate. It is important to keep that last part in mind, since this data set only represents loans actually made, i.e. do not mistake this data for loan applications!

## Ojective:
- Describe the dataset and any issues with it.
- Generate a minimum of 5 unique visualizations using the data and write a brief description of your observations. Additionally, all attempts should be made to make the visualizations visually appealing
- Create a feature set and create a model which predicts interest rate using at least 2 algorithms. Describe any data cleansing that must be performed and analysis when examining the data.
- Visualize the test results and propose enhancements to the model, what would you do if you had more time. Also describe assumptions you made and your approach.

- Source
 - This data comes from Lending Club (https://www.lendingclub.com/info/statistics.action), which provides a very large, open set of data on the people who received loans through their platform.
 - 


## Walk-through of the Project
- 1. Cleansing, Preprocessing and EDA
    - Look at missing values
    - Distribution of interes rate
    - Categorical Variables
        -Explore categorical variables and interest rate
    - Numerical Variables
        -Explore numerical variables and interest rate
- 2. Feature engineering 
    - Adding more variables
    - Scaling & Getting dummy
    - Feature selection(Lasso CV)
- 3. Model
    - Random Forest
    - XGBoost

## Conclusion
- EDA
    - 10000 sample size with 55columns.
    - Many variables containing outliers and missing values
    - Interest rate distribution are right-skewed. If we use linear regression, we should log-transform the interest rate
    - Grades an subgrades are highly correlated to interest rate
- Model Selection
    - Randomforest Model has mean MAE-0.383 and MAPE 3.55
    - XGBoost Model has mean MAE -0.491 and MAPE 4.64
    - Randomforest would be a better choice
- Feature Selection
    - The feature I choose are basily about Grade and Subgrade
        -grade: Grade associated with the loan.
        -sub_grade: Detailed grade associated with the loan.
    - However, we don't know what does grade are given. Only when we find out what influence grades, we can deep dive into different variables that affecting interest rate.
- Next step:
    - Add more models(Neural Networks and Linear regression)
    - Explore more about how does grades and sub-grades influences the interest rate. Correlaiton does not mean causual inferences
    - Explore more on the parameters,optimizing the performance of the model
