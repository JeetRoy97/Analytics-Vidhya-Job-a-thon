# Analytics-Vidhya-Job-a-thon

# Problem Statement

ABC is a car rental company based out of Bangalore. It rents cars for both in and out stations at affordable prices. The users can rent different types of cars like Sedans, Hatchbacks, SUVs and MUVs, Minivans and so on.

In recent times, the demand for cars is on the rise. As a result, the company would like to tackle the problem of supply and demand. The ultimate goal of the company is to strike the balance between the supply and demand inorder to meet the user expectations. The company has collected the details of each rental. Based on the past data, the company would like to forecast the demand of car rentals on an hourly basis.

The train set consists of data from 2018-08-18 to 2021-02-28 	. The dataset consists of date, hour and demand. On the basis of date and hour, the task is to predict the demand from 2021-03-01 to 2022-03-28.

Since, in our first step, to create the feature set, we extracted the relevant information such as year, month, quarter_of_year, week_of_year, day_of_year, day_of_month,  day_of_week, is_wknd, is_month_start, is_month_end. Along with this, we also added some random noise and time shift lags to make the forecasting more effective.

In the next step, we calculated roll mean features and ewm features with specific alpha and lag parameters. This way the feature composed of 107 features. The train set is subdivided into train dataframe ranging from 2018-08-18 to 2021-01-01 and validation dataframe from 2021-01-01 to 2022-02-28. The train df shape become (16942, 107) whereas the val df shape become (8341, 107).

In the next step, we initialized the LGBM parameters {'num_leaves': 10, 'learning_rate': 0.02, 'feature_fraction': 0.8,'max_depth': 5,'verbose': 0,'num_boost_round': 10000,'early_stopping_rounds': 200,'nthread': -1}
