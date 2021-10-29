#!/usr/bin/env python
# coding: utf-8

# ## Lending Club Interest Rate Prediction

# - Description
#  - This data set represents thousands of loans made through the Lending Club platform, which is a platform that allows individuals to lend to other individuals. Of course, not all loans are created equal. Someone who is a essentially a sure bet to pay back a loan will have an easier time getting a loan with a low interest rate than someone who appears to be riskier. And for people who are very risky? They may not even get a loan offer, or they may not have accepted the loan offer due to a high interest rate. It is important to keep that last part in mind, since this data set only represents loans actually made, i.e. do not mistake this data for loan applications!
# 
# - Source
#  - This data comes from Lending Club (https://www.lendingclub.com/info/statistics.action), which provides a very large, open set of data on the people who received loans through their platform.

# ## Walk-through of the Project
# - 1. Cleansing, Preprocessing and EDA
#     - Look at missing values
#     - Distribution of interes rate
#     - Categorical Variables
#         -Explore categorical variables and interest rate
#     - Numerical Variables
#         -Explore numerical variables and interest rate
# - 2. Feature engineering 
#     - Adding more variables
#     - Scaling & Getting dummy
#     - Feature selection(Lasso CV)
# - 3. Model 
#     - Random Forest
#     - XGBoost

# ## Import data

# In[155]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from string import ascii_letters
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv("loans_full_schema.csv")


# In[3]:


data.head()


# ## Cleansing, Preprocessing and EDA

# In[4]:


data.info()


# 10000 sample size with 55columns.

# In[5]:


df_miss=data.isnull().sum()/len(data)*100
df_miss=pd.DataFrame(df_miss,columns=['percentage'])


# Difficult for to fill in the null values because emp_title are categorical vairables.Too much category.Just delete this colum

# In[6]:


##delete missing data>50%
df_miss[df_miss['percentage']>50]


# In[7]:


##drop vairables missing values percentage>50
data_new=data.drop(df_miss[df_miss['percentage']>50].index,axis=1)


# In[8]:


data_new.columns


# ### Dealing with categorical variables

# In[9]:


cat_cols = data_new.select_dtypes(include=("object"))


# In[10]:


cat_cols.columns


# In[11]:


cat_cols['emp_title'].nunique()


# In[12]:


cat_cols['state'].nunique()


# In[13]:


cat_cols['emp_title'].value_counts()


# In[14]:


## Too many unique values for employment title, which is low
cat_cols=cat_cols.drop(['emp_title'],axis=1)


# In[15]:


for i in cat_cols.columns.tolist():
    print(data_new[i].value_counts())


# In[16]:


from scipy.stats import spearmanr


# In[17]:


## Not much correlaiton between state and interest rate
data_new['state'].corr(data_new['interest_rate'],method='spearman')


# In[18]:


cat_cols=cat_cols.drop(['state'],axis=1)


# the correlation is not strong and delete the state columns

# In[19]:


for i in cat_cols.columns.tolist():
    cor=data_new[i].corr(data_new['interest_rate'],method='spearman')
    print(i,cor)


# In[222]:


for i in cat_cols.columns.tolist():
    cor=data_new[i].corr(data_new['grade'],method='spearman')
    print(i,cor)


# In[21]:


data_new['interest_rate'].round().value_counts()


# In[224]:


data_new.describe()


# Many variables containing outliers and missing values

# ### Distribution of interest rate

# In[22]:


plt.figure(figsize=(12,6))
plt.subplot(121)
g = sns.distplot(data_new["interest_rate"])
g.set_xlabel("interest_rate", fontsize=12)
g.set_ylabel("Frequency Dist", fontsize=12)
g.set_title("Frequency Distribuition", fontsize=20)
plt.subplot(122)
g1 = sns.violinplot(y="interest_rate", data=data_new, 
               inner="quartile", palette="hls")
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Amount Dist", fontsize=12)
g1.set_title("Amount Distribuition", fontsize=20)

plt.show()


# In[24]:


## The distribution of interest rate is right-skewed, we should log transfrom the interest rate
#Exploring the Int_rate
data_new['log_int_rate']=np.log(data_new["interest_rate"])
plt.figure(figsize=(12,6))
plt.subplot(211)
g = sns.distplot(data_new['log_int_rate'])
g.set_xlabel("", fontsize=12)
g.set_ylabel("Distribuition", fontsize=12)
g.set_title("Int Rate Log distribuition", fontsize=20)
plt.show()


# If we are transform the distribution of interest rate, we can apply linear regression

# ###  Explore categorical variables and interest rate

# ### Grades and Subgrades

# In[96]:


##Grade& interest rate
plt.figure(figsize=(12,12))
plt.subplot(211)
sns.boxplot(x="grade", y="interest_rate", data=data_new,palette="hls", hue="application_type", order=["A",'B','C','D','E','F', 'G'])
plt.subplot(212)
sns.boxenplot(x="sub_grade", y="interest_rate", data=data_new,palette="hls")


# It is obvious that grade had high correlation with interest rate
# Higher grade, higher interest rate

# In[25]:


##Loan_status& interest rate
sns.displot(data=data_new,x="interest_rate",hue='loan_status',kind='kde')


# In[28]:


##Homeownership& interest rate
data_new['homeownership'].value_counts()


# In[30]:


data_new['homeownership_rent']=np.where(data_new['homeownership']=='RENT',1,0)


# In[33]:


fig= plt.figsize=(24,10)
sns.boxplot(x="homeownership_rent", y="interest_rate", data=data_new,palette="hls",order=[1,0])


# In[34]:


##Homeownership& interest rate
data_new['verified_income'].value_counts()


# In[35]:


plt.figure(figsize = (10,6))

g = sns.violinplot(x="verified_income",y="interest_rate",data=data_new,
               kind="violin",
               split=True,palette="hls")
g.set_title("Verified_income - interest_rate", fontsize=20)
g.set_xlabel("", fontsize=15)
g.set_ylabel("Interest Rate", fontsize=15)

plt.show()


# higher correlation with grade, maintain the columns

# All categorical variables are bearable number of categories

# In[72]:


df_cat = pd.get_dummies(cat_cols,columns=cat_cols.columns.tolist(),drop_first=True)


# df_cat

# ## Numerical columns processing

# In[39]:


# Subset numeric features: numeric_cols
numeric_cols = data_new.select_dtypes(include=[np.number])
# Iteratively impute
imp_iter = IterativeImputer(max_iter=5, sample_posterior=True, random_state=123)
loans_imp_iter = imp_iter.fit_transform(numeric_cols)
# Convert returned array to DataFrame
loans_imp_iterDF = pd.DataFrame(loans_imp_iter, columns=numeric_cols.columns)


# In[40]:


numeric_cols.columns


# In[41]:


##correlations
sns.set(style="white")

corr = loans_imp_iterDF.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[81]:


X_num=loans_imp_iterDF.drop(['interest_rate'],axis=1)
for i in X_num.columns.tolist():
    cor=loans_imp_iterDF[i].corr(loans_imp_iterDF['interest_rate'])
    print(i,cor)


# In[43]:


plt.plot(data_new['term'],data_new['interest_rate'],'bo')
plt.show()


# In[45]:


data_new['term_36']=data_new['term'][data_new['term']==36]
data_new['term_60']=data_new['term'][data_new['term']==60]


# In[46]:


data_new['term_36']=np.where(data_new['term_36']==36,1,0)
data_new['term_60']=np.where(data_new['term_60']==60,1,0)


# In[47]:


loans_imp_iterDF=loans_imp_iterDF.drop(['term'],axis=1)


# In[221]:


plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(data_new['debt_to_income'],data_new['interest_rate'],'bo')
plt.title('debt_to_income ~ interest_rate')
plt.subplot(122)
plt.plot(data_new['annual_income'],data_new['interest_rate'],'g*')
plt.title('annual_income ~ interest_rate')
plt.show()


# In[82]:


df_for_RF=pd.concat([df_cat,loans_imp_iterDF],axis=1)


# In[83]:


len(df_for_RF.columns)


# In[84]:


df_total['log_int_rate'].isnull().sum()


# df_total.to_csv('df_total.csv')

# ## DATA Scaling

# In[127]:


#normalise the data
X_train, y_train = Train.drop(['log_int_rate','interest_rate'], axis=1), Train.interest_rate
X_test, y_test = Test.drop(['log_int_rate','interest_rate'], axis=1), Test.interest_rate
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
df_for_RF=pd.merge(X_train,)


# ## Feature selection
# Too much variables for 10000 samples

# In[ ]:


df_for_RF


# In[90]:


from sklearn.ensemble import RandomForestClassifier


# In[101]:


df_for_RF.info()


# In[109]:


size = df_for_RF.shape[0]

Train, Test_new = train_test_split(df_for_RF, test_size= 0.3, random_state= 1)

CV, Test_new = train_test_split(Test_new, test_size=0.5, random_state = 1)

print(Train.shape, CV.shape, Test_new.shape)


# In[115]:


CV


# In[133]:


X_train, y_train = Train.drop(['log_int_rate','interest_rate'], axis=1), Train.interest_rate
X_test, y_test = Test_new.drop(['log_int_rate','interest_rate'], axis=1), Test.interest_rate
CV_x,CV_y= CV.drop(['log_int_rate','interest_rate'], axis=1), CV.interest_rate
x_col=X_train.columns


# In[134]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
CV_x=scaler.transform(CV_x)


# In[135]:


Test.info()


# In[136]:


CV_x


# In[137]:


from sklearn.linear_model import LassoCV


# In[140]:


modellasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0001, 10, 1000]).fit(X_train,y_train)
lassopred = modellasso.predict(CV_x)
print("RMSE of Lasso: ", np.sqrt(mean_squared_error(lassopred, CV_y)))

coeff = modellasso.coef_

x = list(x_col)
x_pos = [i for i, _ in enumerate(x)]


plt.figure(figsize = (10,40))
plt.barh(x_pos, coeff, color='green')
plt.ylabel("Features -->")
plt.xlabel("Coefficents -->")
plt.title("Coefficents from Lasso")
plt.yticks(x_pos, x)

plt.show()


# In[142]:


data_col=pd.DataFrame(columns=['name','coeff'])
co=list(coeff)


# In[143]:


data_col['name']=x_col
data_col['coeff']=co


# In[144]:


data_col.sort_values(by='coeff',ascending=False)


# In[145]:


data_col.count()


# In[183]:


col=data_col[abs(data_col['coeff'])>0.5]


# In[184]:


col.count()


# In[185]:


col['name'].tolist()


# ## Random Forest
# - Advantages of random forest
#     - It can perform both regression and classification tasks.
#     - A random forest produces good predictions that can be understood easily.
#     - It can handle large datasets efficiently.
#     - The random forest algorithm provides a higher level of accuracy in predicting outcomes over the decision tree algorithm.
# - Disadvantages of random forest
#     - When using a random forest, more resources are required for computation.
#     - It consumes more time compared to a decision tree algorithm.

# In[186]:


features= df_for_RF[col['name'].tolist()]
labels=df_total['interest_rate']


# In[187]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3,random_state = 42)


# In[188]:


print('Training Features Shape:', train_features.shape)
print('Testing Features Shape:', test_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[189]:


from sklearn.ensemble import RandomForestRegressor


# In[209]:


# Instantiate model (Using Default 10 Estimators)
rf = RandomForestRegressor(n_estimators= 10, random_state=42)

# Using Evaluation Function on our First Model

rf.fit(train_features, train_labels)

y_test_pred = rf.predict(test_features)


# In[210]:


y_test_pred.shape


# In[211]:


# Mean squared error
from sklearn import metrics
print("Mean squared error: %.2f" % mean_squared_error(test_labels, y_test_pred))
print('Mean Absolute Percentage Error (MAPE):', round(metrics.mean_absolute_percentage_error(test_labels, y_test_pred) * 100, 2))
print('Accuracy:', round(100*(1 - metrics.mean_absolute_percentage_error(test_labels, y_test_pred)), 2))


# ## Randomforest tuning
# For model performance improvement, We should use parameter tunning
# - n_estimators
#   - The n_estimators parameter specifies the number of trees in the forest of the model. The default value for this parameter is 10, which means that 10 different decision trees will be constructed in the random forest.
# - max_depth 
#   - The max_depth parameter specifies the maximum depth of each tree. The default value for max_depth is None, which means that each tree will expand until every leaf is pure. A pure leaf is one where all of the data on the leaf comes from the same class.
# - min_samples_split
#    - The min_samples_split parameter specifies the minimum number of samples required to split an internal leaf node. The default value for this parameter is 2, which means that an internal node must have at least two samples before it can be split to have a more specific classification.
# - min_samples_leaf
#    - The min_samples_leaf parameter specifies the minimum number of samples required to be at a leaf node. The default value for this parameter is 1, which means that every leaf must have at least 1 sample that it classifies.

# *
# param_test1 = {'n_estimators':range(10,71,10),
#               {'max_depth':range(3,14,2),
#               'min_samples_split':range(50,201,20),
#               'min_samples_leaf':range(30,60,10)
#               }
# gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100
# min_samples_leaf=20,max_depth=8,max_features='
# param_grid = param_test1, scoring='roc_auc',cv=5)
# gsearch1.fit(X,y)
# gsearch1.best_params_, gsearch1.best_score_
# *

# ## XGBoost
# XGBoost is a highly optimized framework for gradient boosting, an algorithm that iteratively combines the predictions of several weak learners such as decision trees to produce a much stronger and more robust model.

# In[193]:


from scipy import stats 
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc,
    plot_confusion_matrix, plot_roc_curve
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier


# param_grid = dict(
#     n_estimators=stats.randint(10, 500),
#     max_depth=stats.randint(1, 10),
#     learning_rate=stats.uniform(0, 1)
#     )
# 
# xgb_clf = XGBClassifier()
# xgb_cv = RandomizedSearchCV(
#     xgb_clf, param_grid, cv=3, n_iter=60, 
#     scoring='roc_auc', n_jobs=-1, verbose=1)
# xgb_cv.fit(X_train, y_train)

# best_params = xgb_cv.best_params_
# print(best_params)

# In[175]:


best_params['booster'] = 'gblinear'
print(f"Best Parameters: {best_params}")


# In[212]:


xgb_clf = XGBClassifier({'learning_rate': 0.0012592383320760847, 'max_depth': 8, 'n_estimators': 204, 'booster': 'gblinear'})


xgb_clf.fit(train_features, train_labels)

y_test_pred = xgb_clf.predict(test_features)


# In[213]:


print("Mean squared error: %.2f" % mean_squared_error(test_labels, y_test_pred))
print('Mean Absolute Percentage Error (MAPE):', round(metrics.mean_absolute_percentage_error(test_labels, y_test_pred) * 100, 2))
print('Accuracy:', round(100*(1 - metrics.mean_absolute_percentage_error(test_labels, y_test_pred)), 2))


# In[204]:


# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(xgb_clf, train_features, train_labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )


# In[203]:


# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(rf, train_features, train_labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )


# ## Conclusion
# 

# - EDA
#     - 10000 sample size with 55columns.
#     - Many variables containing outliers and missing values
#     - many variables are high-imbalanced.
#     - Interest rate distribution are right-skewed. If we use linear regression, we should log-transform the interest rate
#     - Grades an subgrades are highly correlated to interest rate
# - Model Selection
#     - Randomforest Model has mean MAE-0.383 and MAPE 3.55
#     - XGBoost Model has mean MAE -0.491 and MAPE 4.64
#     - Randomforest would be a better choice
# - Feature Selection
#     - The feature I choose are basily about Grade and Subgrade
#         -grade: Grade associated with the loan.
#         -sub_grade: Detailed grade associated with the loan.
#     - However, we don't know what does grade are given. Only when we find out what influence grades, we can deep dive into different variables that affecting interest rate.
# - Next step:
#     - Add more models(Neural Networks and Linear regression)
#     - Explore more about how does grades and sub-grades influences the interest rate. Correlaiton does not mean causual inferences
#     - Explore more on the parameters,optimizing the performance of the model

# In[ ]:




