Model information
So for making a prediction we picked 3 models, 
1.Linear Regression
2.randomForest
3.Xgboost

Target variable and depending variables (x,y)
so our target variable is Datavalue
we particularly picked number category from the datavaluetype and started working on them as we thought this number in datavalue category can clearly explain instead of working on mixed populations like crude prelavance,age adjusted number which deals with % and in second step out of 17 topic i picked Cardiovascular disease as i found out that topic cardiovascular has highest occurence after filtering the data as datavalue(numbr)->topic(Cardiovascular Disease) after finding out out total poulation its evident that i can make a prediction on that so for my dependants im taking (yearstart,topic,question,stratificationid,locationabbr) as my dependants and target variable is datavalue, so i split my train and test data as 70/30 and tested that on existing data which i got very good number almost 6000 away but still i can justify because the highest datavalues that i saw were almost close to milllion and mse values range from 0~ infinity,and after fully training the data i got pretty good values with randomForest and Xgboost, i predicted for year 2023 and i found the values are pretty close and i see the trend of decreasing (as we are advancing in terms of developing medicines and cures for the disease and i see a propotionality decrease) like i checked with the data of last year which is 2020 and 2o21 to 2023.

Model building
As i mentioned earlier about my dependant variable X which has both numerical and categorical variable (Ex:locationabbr)so i had to onehot encode them and so i got further extra 51 rows and other dependant variable as well at the end i have 96 features so my training set has both num+cat variables, generalised train and test split 70/30, Models =Linear regression(),randomForest(),Xgboost, further i had to map back the encoded values to state codes to eliminate confusion while asking for user iput for state to know the prediction for Cardiovascular Cases.

Metrics and evaluation
I considered Rmse and r2 values
Linear Regression - RMSE: 20539.89292783556, R2: 0.6036012367259129
Random Forest - RMSE: 2345.221197407292, R2: 0.9948322254117209
Chosen model XGBoost - RMSE: 1677.9872758773472, R2: 0.9973544664914713
So i picked XGboost as my model as it gave pretty good values and when i compared the values with last years of data in dataset i see propotionality decrease, and the metrics are also good


Prediction:
So the model predicts total no.of Cardiovascular Disease Cases for every state in US
taking the state code as input
