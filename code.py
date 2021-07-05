

#pip install scikit-surprise                     install the surprise library in python

                                       #####  Colaborative based Recommendation system  ######

import pandas as pd

from surprise import SVD
import numpy as np
import surprise
from surprise import Reader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

from datetime import datetime

data = pd.read_csv("/content/drive/MyDrive/ml-latest-small/ratings.csv")
xx=pd.read_csv("/content/drive/MyDrive/ml-latest-small/movies.csv")

data.head()

xx.head()

data.shape

data = data.drop('timestamp',axis=1)

data.head()

#train_data.shape

train_data=data.iloc[:int(data.shape[0]*0.80)]
test_data=data.iloc[int(data.shape[0]*0.80):]

train_data.head()

test_data.shape

#################### SVD ############################





#####################################################


########################################Trainset into Surprise library################################


# It is to specify how to read the dataframe.
# for our dataframe, we don't have to specify anything extra..
reader = Reader(rating_scale=(1,5))

# create the traindata from the dataframe...
train_data_mf = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)

# build the trainset from traindata.., It is of dataset format from surprise library..
trainset = train_data_mf.build_full_trainset()

########################################Testset into Surprise library#####################################


# It is to specify how to read the dataframe.
# for our dataframe, we don't have to specify anything extra..
reader = Reader(rating_scale=(1,5))

# create the traindata from the dataframe...
test_data_mf = Dataset.load_from_df(test_data[['userId', 'movieId', 'rating']], reader)

# build the trainset from traindata.., It is of dataset format from surprise library..
testset = test_data_mf.build_full_trainset()

#############################Implementing the Model######################################

svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
svd.fit(trainset)



#getting predictions of trainset
train_preds = svd.test(trainset.build_testset())

train_pred_mf = np.array([pred.est for pred in train_preds])

def get_error_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean([ (y_true[i] - y_pred[i])**2 for i in range(len(y_pred)) ]))
    mape = np.mean(np.abs( (y_true - y_pred)/y_true )) * 100
    return rmse, mape

train_pred=pd.DataFrame(train_pred_mf)
train_pred.head()

train_data['rating'].head()
train_actual_rating=train_data['rating']
train_actual_rating.shape
train_actual=pd.DataFrame(train_actual_rating)
train_actual.shape


#getting predictions of trainset
test_preds = svd.test(testset.build_testset())

test_pred_mf = np.array([pred.est for pred in test_preds])

test_pred=pd.DataFrame(test_pred_mf)
test_pred.head()
test_pred.shape

test_actual_rating=test_data['rating']


test_actual_rating.shape
test_actual=pd.DataFrame(test_actual_rating)
test_actual.shape

#################  Error Analysis  ################


##Train##
from sklearn.metrics import mean_squared_error
train_actual
rmse_train = mean_squared_error( train_pred,train_actual, squared=False)
rmse_train

"""
rmse_train, mape_train = get_error_metrics(train_actual, train_pred)
    
# store the results in train_results dictionary..
train_results = {'rmse': rmse_train,
                    'mape' : mape_train,
                    'predictions' : train_pred}
"""                    



##Test##
from sklearn.metrics import mean_squared_error

rmse_test = mean_squared_error( test_pred,test_actual, squared=False)
rmse_test
"""
  ############################################### Preparing for Regression Mopdel ###########################################
To improve the results of our model we focussed on some collaborative aspects of the moive ratings.
Now we are trying to implementing Regression models of machine learning earlier we used matrix factoristion method called SVD(Singular value Decomposition)

To include the collaborative aspects into our model we created extra features to our model using "cosine_similarity"
we added features corresponding to top 5 similar users corresponding to a user and also added top 5 similar movies corresponding to a similar movies for a particular movies.
we also added features for user average and movie averages as well.

And implemented various linear rehgression and test our performance. 
"""

                                                       ######  XGBoost #######

# Creating a sparse matrix
from scipy.sparse import csr_matrix
from scipy import sparse

# Creating a sparse matrix
train_sparse_matrix = sparse.csr_matrix((train_data.rating.values, (train_data.userId.values,
                                               train_data.movieId.values)))

# Global avg of all movies by all users
train_sparse_matrix.shape

train_averages = dict()
# get the global average of ratings in our train set.
train_global_average = train_sparse_matrix.sum()/train_sparse_matrix.count_nonzero()
train_averages['global'] = train_global_average
train_averages


# get the user averages in dictionary (key: user_id/movie_id, value: avg rating)

def get_average_ratings(sparse_matrix, of_users):
    
    # average ratings of user/axes
    ax = 1 if of_users else 0 # 1 - User axes,0 - Movie axes

    # ".A1" is for converting Column_Matrix to 1-D numpy array 
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    # Boolean matrix of ratings ( whether a user rated that movie or not)
    is_rated = sparse_matrix!=0
    # no of ratings that each user OR movie..
    no_of_ratings = is_rated.sum(axis=ax).A1
    
    # max_user  and max_movie ids in sparse matrix 
    u,m = sparse_matrix.shape
    # creae a dictonary of users and their average ratigns..
    average_ratings = { i : sum_of_ratings[i]/no_of_ratings[i]
                                 for i in range(u if of_users else m) 
                                    if no_of_ratings[i] !=0}

    # return that dictionary of average ratings
    return average_ratings

# Average ratings given by a user


train_averages['user'] = get_average_ratings(train_sparse_matrix, of_users=True)
print('\nAverage rating of user 10 :',train_averages['user'][10])

# Average ratings given for a movie

train_averages['movie'] =  get_average_ratings(train_sparse_matrix, of_users=False)
print('\n AVerage rating of movie 15 :',train_averages['movie'][15])

# get users, movies and ratings from our samples train sparse matrix
train_users, train_movies, train_ratings = sparse.find(train_sparse_matrix)

final_data = pd.DataFrame()
count = 0
for (user, movie, rating)  in zip(train_users, train_movies, train_ratings):
            st = datetime.now()
        #     print(user, movie)    
            #--------------------- Ratings of "movie" by similar users of "user" ---------------------
            # compute the similar Users of the "user"        
            user_sim = cosine_similarity(train_sparse_matrix[user], train_sparse_matrix).ravel()
            top_sim_users = user_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
            # get the ratings of most similar users for this movie
            top_ratings = train_sparse_matrix[top_sim_users, movie].toarray().ravel()
            # we will make it's length "5" by adding movie averages to .
            top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
            top_sim_users_ratings.extend([train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))
        #     print(top_sim_users_ratings, end=" ")    


            #--------------------- Ratings by "user"  to similar movies of "movie" ---------------------
            # compute the similar movies of the "movie"        
            movie_sim = cosine_similarity(train_sparse_matrix[:,movie].T, train_sparse_matrix.T).ravel()
            top_sim_movies = movie_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
            # get the ratings of most similar movie rated by this user..
            top_ratings = train_sparse_matrix[user, top_sim_movies].toarray().ravel()
            # we will make it's length "5" by adding user averages to.
            top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])
            top_sim_movies_ratings.extend([train_averages['user'][user]]*(5-len(top_sim_movies_ratings))) 
        #     print(top_sim_movies_ratings, end=" : -- ")

            #-----------------prepare the row to be stores in a file-----------------#
            row = list()
            row.append(user)                                                    
            row.append(movie)
            # Now add the other features to this data...
            row.append(train_averages['global']) # first feature
            # next 5 features are similar_users "movie" ratings
            row.extend(top_sim_users_ratings)
            # next 5 features are "user" ratings for similar_movies
            row.extend(top_sim_movies_ratings)
            # Avg_user rating
            row.append(train_averages['user'][user])                                                             
            # Avg_movie rating
            row.append(train_averages['movie'][movie])

            # finalley, The actual Rating of this user-movie pair...
            row.append(rating)
            count = count + 1
            final_data = final_data.append([row])
           # print(count)

           
        
            if (count)%1000 == 0:
                print("Done for {} rows----- {}".format(count, datetime.now() - st ))
               

final_data.columns=['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5',
            'smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating']


final_data.head()

############################Adding predictions from SVD as a feature###########################

##final_data['mf_svd']=train_pred_mf

final_data.head()

############  Preparing Test Data  ######################




#############################################################

# Creating a sparse matrix
test_sparse_matrix = sparse.csr_matrix((test_data.rating.values, (test_data.userId.values,
                                               test_data.movieId.values)))

# Global avg of all movies by all users

test_averages = dict()
# get the global average of ratings in our train set.
test_global_average = test_sparse_matrix.sum()/test_sparse_matrix.count_nonzero()
test_averages['global'] = test_global_average
test_averages

# get the user averages in dictionary (key: user_id/movie_id, value: avg rating)

def get_average_ratings(sparse_matrix, of_users):
    
    # average ratings of user/axes
    ax = 1 if of_users else 0 # 1 - User axes,0 - Movie axes

    # ".A1" is for converting Column_Matrix to 1-D numpy array 
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    # Boolean matrix of ratings ( whether a user rated that movie or not)
    is_rated = sparse_matrix!=0
    # no of ratings that each user OR movie..
    no_of_ratings = is_rated.sum(axis=ax).A1
    
    # max_user  and max_movie ids in sparse matrix 
    u,m = sparse_matrix.shape
    # creae a dictonary of users and their average ratigns..
    average_ratings = { i : sum_of_ratings[i]/no_of_ratings[i]
                                 for i in range(u if of_users else m) 
                                    if no_of_ratings[i] !=0}

    # return that dictionary of average ratings
    return average_ratings

# Average ratings given by a user

test_averages['user'] = get_average_ratings(test_sparse_matrix, of_users=True)
#print('\nAverage rating of user 10 :',test_averages['user'][10])

# Average ratings given for a movie

test_averages['movie'] =  get_average_ratings(test_sparse_matrix, of_users=False)
print('\n AVerage rating of movie 15 :',test_averages['movie'][15])


# get users, movies and ratings from our samples train sparse matrix
test_users, test_movies, test_ratings = sparse.find(test_sparse_matrix)



final_test_data = pd.DataFrame()
count = 0
for (user, movie, rating)  in zip(test_users, test_movies, test_ratings):
            st = datetime.now()
        #     print(user, movie)    
            #--------------------- Ratings of "movie" by similar users of "user" ---------------------
            # compute the similar Users of the "user"        
            user_sim = cosine_similarity(test_sparse_matrix[user], test_sparse_matrix).ravel()
            top_sim_users = user_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
            # get the ratings of most similar users for this movie
            top_ratings = test_sparse_matrix[top_sim_users, movie].toarray().ravel()
            # we will make it's length "5" by adding movie averages to .
            top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
            top_sim_users_ratings.extend([test_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))
        #     print(top_sim_users_ratings, end=" ")    


            #--------------------- Ratings by "user"  to similar movies of "movie" ---------------------
            # compute the similar movies of the "movie"        
            movie_sim = cosine_similarity(test_sparse_matrix[:,movie].T, test_sparse_matrix.T).ravel()
            top_sim_movies = movie_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
            # get the ratings of most similar movie rated by this user..
            top_ratings = test_sparse_matrix[user, top_sim_movies].toarray().ravel()
            # we will make it's length "5" by adding user averages to.
            top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])
            top_sim_movies_ratings.extend([test_averages['user'][user]]*(5-len(top_sim_movies_ratings))) 
        #     print(top_sim_movies_ratings, end=" : -- ")

            #-----------------prepare the row to be stores in a file-----------------#
            row = list()
            row.append(user)
            row.append(movie)
            # Now add the other features to this data...
            row.append(test_averages['global']) # first feature
            # next 5 features are similar_users "movie" ratings
            row.extend(top_sim_users_ratings)
            # next 5 features are "user" ratings for similar_movies
            row.extend(top_sim_movies_ratings)
            # Avg_user rating
            row.append(test_averages['user'][user])
            # Avg_movie rating
            row.append(test_averages['movie'][movie])

            # finalley, The actual Rating of this user-movie pair...
            row.append(rating)
            count = count + 1
            final_test_data = final_test_data.append([row])
            #print(count)

           
        
            if (count)%1000 == 0:
                # print(','.join(map(str, row)))
                print("Done for {} rows----- {}".format(count, datetime.now() - st))

final_test_data.columns=['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5',
            'smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating']

final_test_data.head()

final_data.shape
final_test_data.shape

##final_data = final_data.drop('mf_svd',axis=1)

############################################## XGBoost ####################################################################


def get_error_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean([ (y_true[i] - y_pred[i])**2 for i in range(len(y_pred)) ]))
    mape = np.mean(np.abs( (y_true - y_pred)/y_true )) * 100
    return rmse, mape

# prepare train data
x_train = final_data.drop(['user', 'movie','rating'], axis=1)
y_train = final_data['rating']


# Prepare Test data
x_test = final_test_data.drop(['user','movie','rating'], axis=1)
y_test = final_test_data['rating']


import xgboost as xgb

# initialize XGBoost model...
xgb_model = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15, n_estimators=100)
# dictionaries for storing train and test results
train_results = dict()
test_results = dict()
    
    
# fit the model
print('Training the model..')
start =datetime.now()
xgb_model.fit(x_train, y_train, eval_metric = 'rmse')
print('Done. Time taken : {}\n'.format(datetime.now()-start))
print('Done \n')

# from the trained model, get the predictions....
print('Evaluating the model with TRAIN data...')
start =datetime.now()
y_train_pred = xgb_model.predict(x_train)
# get the rmse and mape of train data...
rmse_train, mape_train = get_error_metrics(y_train.values, y_train_pred)
    
# store the results in train_results dictionary..
train_results = {'rmse': rmse_train,
                    'mape' : mape_train,
                    'predictions' : y_train_pred}

train_results

########################################## Test Data Predictions ##################################################
# get the test data predictions and compute rmse and mape
print('Evaluating Test data')
y_test_pred = xgb_model.predict(x_test) 
rmse_test, mape_test = get_error_metrics(y_true=y_test.values, y_pred=y_test_pred)
# store them in our test results dictionary.
test_results = {'rmse': rmse_test,
                    'mape' : mape_test,
                    'predictions':y_test_pred}


test_results

These were the results on applying XGBoost
Now we are using SVD and XGBoost combined 

########################## Adding SVD results for better results ###########################


###############  XGBoost + SVD  ########################





########################################################

final_data['mf_svd']=train_pred_mf


final_test_data['mf_svd']=test_pred_mf

# prepare train data
x_train = final_data.drop(['user', 'movie','rating'], axis=1)
y_train = final_data['rating']

# Prepare Test data
x_test = final_test_data.drop(['user','movie','rating'], axis=1)
y_test = final_test_data['rating']

x_train.head()


# initialize XGBoost model...
xgb_model = xgb.XGBRegressor(silent=False, n_jobs=15, random_state=15, n_estimators=150)
# dictionaries for storing train and test results
train_results = dict()
test_results = dict()
    
    
# fit the model
print('Training the model..')
start =datetime.now()
xgb_model.fit(x_train, y_train, eval_metric = 'rmse')
print('Done. Time taken : {}\n'.format(datetime.now()-start))
print('Done \n')

# from the trained model, get the predictions....
print('Evaluating the model with TRAIN data...')
start =datetime.now()
y_train_pred = xgb_model.predict(x_train)
# get the rmse and mape of train data...
rmse_train, mape_train = get_error_metrics(y_train.values, y_train_pred)
    
# store the results in train_results dictionary..
train_results = {'rmse': rmse_train,
                    'mape' : mape_train,
                    'predictions' : y_train_pred}

train_results

#######################################
# get the test data predictions and compute rmse and mape
print('Evaluating Test data')
y_test_pred = xgb_model.predict(x_test) 
rmse_test, mape_test = get_error_metrics(y_true=y_test.values, y_pred=y_test_pred)
# store them in our test results dictionary.
test_results = {'rmse': rmse_test,
                    'mape' : mape_test,
                    'predictions':y_test_pred}

test_results

final_data.shape


final_test_data.shape


###################### Linear Regression Model #############################




#############################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

Now implementing Linear and Ridge regression algorithm 

x_train

x_train = x_train.drop(['mf_svd'], axis=1)

x_train.head()


y_train


from sklearn.linear_model import LinearRegression, Ridge, Lasso

model = LinearRegression()

# fit the model with the training data
model.fit(x_train,y_train)


######### Prediction on Train set  ######################

# coefficeints of the trained model
print('\nCoefficient of model :', model.coef_)

# intercept of the model
print('\nIntercept of model',model.intercept_)




# predict the target on the test dataset
predict_train = model.predict(x_train)
print('\nItem_Outlet_Sales on training data',predict_train) 

# Root Mean Squared Error on training dataset
rmse_train = mean_squared_error(y_train,predict_train)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)

############# Prediction on Test Dataset #####################



##############################################################


x_test=x_test.drop(['mf_svd'], axis=1)

x_test.head()


# predict the target on the test dataset
predict_test = model.predict(x_test)
print('\nItem_Outlet_Sales on training data',predict_test) 

# Root Mean Squared Error on training dataset
rmse_test = mean_squared_error(y_test,predict_test)**(0.5)
print('\nRMSE on train dataset : ', rmse_test)


import matplotlib.pyplot as plt
predictors = [x for x in x_train.columns]

coef1 = pd.Series(model.coef_,predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')

################# Ridge Regression ########################





###########################################################

alg2 = Ridge(alpha=0.000005,normalize=True)
alg2.fit(x_train,y_train)

# coefficeints of the trained model
print('\nCoefficient of model :', alg2.coef_)

# intercept of the model
print('\nIntercept of model',alg2.intercept_)

# predict the target on the test dataset
predict_train_ridge = alg2.predict(x_train)
print('\nItem_Outlet_Sales on training data',predict_train_ridge) 

# Root Mean Squared Error on training dataset
rmse_train_ridge = mean_squared_error(y_train,predict_train_ridge)**(0.5)
print('\nRMSE on train dataset : ', rmse_train_ridge)

############# Prediction on Test Dataset #####################



##############################################################

# predict the target on the test dataset
predict_test_ridge = alg2.predict(x_test)
print('\nItem_Outlet_Sales on training data',predict_test_ridge) 

# Root Mean Squared Error on training dataset
rmse_test_ridge = mean_squared_error(y_test,predict_test_ridge)**(0.5)
print('\nRMSE on test dataset : ', rmse_test_ridge)

import matplotlib.pyplot as plt
predictors_ridge = [x for x in x_train.columns]

coef_ridge = pd.Series(alg2.coef_,predictors_ridge).sort_values()
coef_ridge.plot(kind='bar', title='Model Coefficients')

final_data

For improving our results adding results from SVD as a feature 

########################################### Ridge regression + SVD ##################################################################

# prepare train data
x_train_g = final_data.drop(['user', 'movie','rating'], axis=1)
y_train_g = final_data['rating']

# Prepare Test data
x_test_g = final_test_data.drop(['user','movie','rating'], axis=1)
y_test_g = final_test_data['rating']

x_train_g.head()

y_train_g.head()

alg2 = Ridge(alpha=0.000005,normalize=True)
alg2.fit(x_train_g,y_train_g)

# coefficeints of the trained model
print('\nCoefficient of model :', alg2.coef_)

# intercept of the model
print('\nIntercept of model',alg2.intercept_)

# predict the target on the test dataset
predict_train_ridge = alg2.predict(x_train_g)
print('\nItem_Outlet_Sales on training data',predict_train_ridge) 

# Root Mean Squared Error on training dataset
rmse_train_ridge = mean_squared_error(y_train_g,predict_train_ridge)**(0.5)
print('\nRMSE on train dataset : ', rmse_train_ridge)

############# Prediction on Test Dataset #####################



##############################################################

# predict the target on the test dataset
predict_test_ridge = alg2.predict(x_test_g)
print('\nItem_Outlet_Sales on training data',predict_test_ridge) 

# Root Mean Squared Error on training dataset
rmse_test_ridge = mean_squared_error(y_test_g,predict_test_ridge)**(0.5)
print('\nRMSE on test dataset : ', rmse_test_ridge)

import matplotlib.pyplot as plt
predictors_ridge = [x for x in x_train_g.columns]

coef_ridge = pd.Series(alg2.coef_,predictors_ridge).sort_values()
coef_ridge.plot(kind='bar', title='Model Coefficients')

################ Error Analysis ##########################

# importing package
import matplotlib.pyplot as plt
import pandas as pd

Error = pd.DataFrame([['SVD', 0.645503,0.998155843], ['Linear Regression', 0.7641402212703406, 0.6907644096118872], ['Ridge Regression', 0.76414022128559330, 0.6904209408818138],
                   ['XGBoost', 0.7532909813, 0.69088280648], ['XGBoost + SVD', 0.749690763790, 0.6870922661127441], ['Ridge Regression + SVD', 0.7641371174747985, 0.6904081732049566] ],
                  columns=['Model', 'Train RMSE', 'Test RMSE'])
# view data
print(Error)
 
# plot grouped bar chart
Error.plot(x='Model',
        kind='bar',
        stacked=False,
        title='Train and Test Errors vs Model')

Error1 = pd.DataFrame([['SVD', 0.645503,0.998155843], ['Linear Regression', 0.7641402212703406, 0.6907644096118872], ['Ridge Regression', 0.76414022128559330, 0.6904209408818138],
                   ['XGBoost', 0.7532909813, 0.69088280648]] ,
                  columns=['Model', 'Train RMSE', 'Test RMSE'])
# view data
print(Error1)
 
# plot grouped bar chart
Error1.plot(x='Model',
        kind='bar',
        stacked=False,
        title='Train and Test Errors vs Model')

Error2 = pd.DataFrame([ ['XGBoost + SVD', 0.749690763790, 0.6870922661127441], ['Ridge + SVD', 0.7641371174747985, 0.6904081732049566] ],
                  columns=['Model', 'Train RMSE', 'Test RMSE'])
# view data
print(Error2)
 
# plot grouped bar chart
Error2.plot(x='Model',
        kind='bar',
        stacked=False,
        title='Train and Test Errors vs Model')

######################################## Exploring the genre column ################################################


########################################      RECOMMENDATION    ####################################################


####################################################################################################################

####################################################################################################################

################################ First Method ############################################



###########################################################################################

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns

movies = pd.read_csv("/content/drive/MyDrive/ml-latest-small/movies.csv")
ratings = pd.read_csv("/content/drive/MyDrive/ml-latest-small/ratings.csv")

ratings.head()

movies.head()


final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.head()
final_dataset.fillna(0,inplace=True)
final_dataset.head()

############## Now lets analyise some data ################

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
no_user_voted

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

f,ax = plt.subplots(1,1,figsize=(16,4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()

################ Improvements can be done #######################

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
final_dataset

f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()

final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
final_dataset

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),\
                               key=lambda x: x[1])[:0:-1]
        
        recommend_frame = []
        
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    
    else:
        
        return "No movies found. Please check your input"

get_movie_recommendation('Guardians of the Galaxy')

get_movie_recommendation('Jumanji')

#References:
#https://analyticsindiamag.com/singular-value-decomposition-svd-application-recommender-system/




############################################# Second Method Not Doing it ####################################################



data2 = pd.read_csv("/content/drive/MyDrive/ml-latest-small/movies.csv")

data2.head()

data2.dtypes

data2.columns





data2['genres'] = data2['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
data2['genres'] = data2['genres'].str.split('|')

data2.head()

import matplotlib.pyplot as plt
import seaborn as sns

plt.subplots(figsize=(12,10))
list1 = []
for i in data2['genres']:
    list1.extend(i)
ax = pd.Series(list1).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',15))
for i, v in enumerate(pd.Series(list1).value_counts()[:15].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Top Genres')
plt.show()

genreList = []
for index, row in data2.iterrows():
    genres = row["genres"]
    
    for genre in genres:
        if genre not in genreList:
            genreList.append(genre)
genreList[:15] #now we have a list with unique genres


data2

#from sklearn.preprocessing import OneHotEncoder

def binary(genre_list):
    binaryList = []
    
    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

#data2['genres_bin'] = data2['genres'].apply(lambda x: binary(x))
#data2['genres_bin'].head(15)

data3 = pd.read_csv("/content/drive/MyDrive/ml-latest-small/tags.csv")

data3=data3.drop(['timestamp','userId'], axis=1)

data3

#data3.groupby('movieId').agg(lambda x: x.tolist())

data3.shape

data2.head()

data2.shape

#import pandas
#dfinal = data2.merge(data3, on="movieId", how = 'inner')


dffinal=pd.merge(data2, data3, on='movieId', how='left').fillna('')

df=dffinal.groupby('movieId').agg(lambda x: x.tolist())

df

df['genres'] = df['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
df['genres'] = df['genres'].str.split('|')



dffinal.groupby('movieId').agg(lambda x: x.tolist())

dffinal['genres'] = dffinal['genres'].str.strip('[]')























