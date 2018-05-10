
# coding: utf-8

# ## Learn model
# ## Content
#   * Load data
#   * Select features
#   * Learning
#     * linear
#     * lasso
#     * ridge
#     * elastic net
#     * Xgboost
#     * MLP

# In[1]:


import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import warnings
import sklearn.linear_model as linear_model

warnings.filterwarnings('ignore')


# In[2]:


MONGODB_URL = os.environ['MONGODB_URI']
client = MongoClient(MONGODB_URL)
db = client.get_default_database()


# In[3]:


data = db["notebook"].find({})
full_frame = pd.DataFrame(list(data))
full_frame.drop(columns=["_id"], inplace=True)
full_frame.shape


# ### Select features

# In[4]:


features = [
    'Neighborhood',
    'GarageFinish',
    'Foundation',
    'MasVnrType',
    'GarageType',
    'MSSubClass',
    'GrLivArea',
    'TotalBsmtSF',
    'GarageCars',
    'BsmtQual',
    'YearBuilt',
    'YearRemodAdd',
    'FireplaceQu',
    'FullBath',
    'BsmtFinSF1',
    'MasVnrArea']

to_log_transform = ['GrLivArea', 'TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1']

to_pow_transform = ['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'Neighborhood', 'GrLivArea']

to_boolean_transform = {
    'TotalBsmtSF': {'new_feature_name': 'HasBasement', 'threshold': 0},
    'GarageArea': {'new_feature_name': 'HasGarage', 'threshold': 0},
    '2ndFlrSF': {'new_feature_name': 'Has2ndFloor', 'threshold': 0},
    'MasVnrArea': {'new_feature_name': 'HasMasVnr', 'threshold': 0},
    'WoodDeckSF': {'new_feature_name': 'HasWoodDeck', 'threshold': 0},
    'OpenPorchSF': {'new_feature_name': 'HasPorch', 'threshold': 0},
    'PoolArea': {'new_feature_name': 'HasPool', 'threshold': 0},
    'YearBuilt': {'new_feature_name': 'IsNew', 'threshold': 2000},
}


# ## Learn model

# In[5]:


def log_transformation(frame, feature):
    new_feature_name = new_log_feature_name(feature)
    frame[new_feature_name] = np.log1p(frame[feature].values)

def new_quadratic_feature_name(feature):
    return feature+'2'

def new_log_feature_name(feature):
    return feature+'Log'
    
def quadratic(frame, feature):
    new_feature_name = new_quadratic_feature_name(feature)
    frame[new_feature_name] = frame[feature]**2
    
def boolean_transformation(frame, feature, new_feature_name, threshold):
    frame[new_feature_name] = frame[feature].apply(lambda x: 1 if x > threshold else 0)
    
def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

def error_mse(actual, predicted):
    actual = (actual)
    predicted = (predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))


# In[6]:


added_boolean_columns = [
    'HasBasement', 
    'HasGarage', 
    'Has2ndFloor', 
    'HasMasVnr', 
    'HasWoodDeck',
    'HasPorch', 
    'HasPool', 
    'IsNew']

added_quadratic_columns = list(map(new_quadratic_feature_name, to_pow_transform))

added_log_columns = list(map(new_log_feature_name, to_log_transform))

def transform_before_learn(frame, to_log_transform, to_pow_transform, to_boolean_transform):

    for c in to_log_transform:
        log_transformation(frame, c)

    for c in to_pow_transform:
        quadratic(frame, c)

    for c in to_boolean_transform.keys():
        boolean_transformation(frame, c, to_boolean_transform[c]['new_feature_name'], 
                               to_boolean_transform[c]['threshold']) 


transform_before_learn(full_frame, to_log_transform, to_pow_transform, to_boolean_transform)

df_train = full_frame[:1460]
df_test = full_frame[1460:]


# In[7]:


features_full_list = features + added_boolean_columns + added_quadratic_columns + added_log_columns


# ## Out liars

# In[8]:


df_train_cleaned = df_train
#df_train_cleaned = df_train.drop(df_train[df_train['Id'] == 1299].index)
#df_train_cleaned = df_train.drop(df_train[df_train['Id'] == 524].index)


# ### LinearRegression

# In[9]:


X = df_train_cleaned[features_full_list]
Y = df_train_cleaned['SalePrice'].values

full_X = df_train[features_full_list]
full_Y = df_train['SalePrice'].values

linear = linear_model.LinearRegression()
linear.fit(X, np.log1p(Y))

Ypred_linear = np.expm1(linear.predict(full_X))
print(error(full_Y, Ypred_linear))
print(error_mse(full_Y, Ypred_linear))


# #### test dataset

# In[10]:


full_test_X = df_test[features_full_list]
test_Y = df_test['SalePrice'].values

test_pred = np.expm1(linear.predict(full_test_X))

print(error(test_Y, test_pred))
print(error_mse(test_Y, test_pred))


# ### LassoCV

# In[11]:


X = df_train_cleaned[features_full_list]
Y = df_train_cleaned['SalePrice'].values

full_X = df_train[features_full_list]
full_Y = df_train['SalePrice'].values

lasso = linear_model.LassoCV()
lasso.fit(X, np.log1p(Y))

Ypred_lasso = np.expm1(lasso.predict(full_X))
print(error(full_Y, Ypred_lasso))
print(error_mse(full_Y, Ypred_linear))


# #### test dataset

# In[12]:


full_test_X = df_test[features_full_list]
test_Y = df_test['SalePrice'].values

test_pred = np.expm1(lasso.predict(full_test_X))

print(error(test_Y, test_pred))
print(error_mse(test_Y, test_pred))


# ### RidgeCV

# In[13]:


X = df_train_cleaned[features_full_list]
Y = df_train_cleaned['SalePrice'].values

full_X = df_train[features_full_list]
full_Y = df_train['SalePrice'].values

ridge = linear_model.RidgeCV()
ridge.fit(X, np.log1p(Y))
Ypred_ridge = np.expm1(ridge.predict(full_X))
print(error(full_Y,Ypred_ridge))
print(error_mse(full_Y, Ypred_ridge))


# #### test dataset

# In[14]:


full_test_X = df_test[features_full_list]
test_Y = df_test['SalePrice'].values

test_pred = np.expm1(ridge.predict(full_test_X))

print(error(test_Y, test_pred))
print(error_mse(test_Y, test_pred))


# ### ElasticNetCV

# In[15]:


X = df_train_cleaned[features_full_list]
Y = df_train_cleaned['SalePrice'].values

full_X = df_train[features_full_list]
full_Y = df_train['SalePrice'].values

elasticNet = linear_model.ElasticNetCV()
elasticNet.fit(X, np.log1p(Y))
Ypred_elasticNet = np.expm1(elasticNet.predict(full_X))
print(error(full_Y,Ypred_elasticNet))
print(error_mse(full_Y, Ypred_elasticNet))


# #### test dataset

# In[16]:


full_test_X = df_test[features_full_list]
test_Y = df_test['SalePrice'].values

test_pred = np.expm1(elasticNet.predict(full_test_X))

print(error(test_Y, test_pred))
print(error_mse(test_Y, test_pred))


# ### Xgboost

# In[17]:


import xgboost as xgb


# In[18]:


X = df_train_cleaned[features_full_list]
Y = df_train_cleaned['SalePrice'].values

full_X = df_train[features_full_list]
full_Y = df_train['SalePrice'].values


# In[19]:


dtrain = xgb.DMatrix(X, label = np.log(Y))

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)


# In[20]:


model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# In[21]:


X_tr, X_val, y_tr, y_val = train_test_split(X, np.log1p(Y), random_state = 42, test_size=0.20)

eval_set = [(X_val, y_val)]


# In[22]:


model_xgb = xgb.XGBRegressor(n_estimators=1000, max_depth=2, learning_rate=0.1)
model_xgb.fit(X_tr, y_tr, eval_metric="rmse", early_stopping_rounds=500, eval_set=eval_set, verbose=True)
# model_xgb.fit(X, np.log1p(Y))


# In[23]:


xgb_preds = np.expm1(model_xgb.predict(full_X))
print(error(full_Y, xgb_preds))
print(error_mse(full_Y, xgb_preds))


# #### test dataset

# In[24]:


full_test_X = df_test[features_full_list]
test_Y = df_test['SalePrice'].values

test_pred = np.expm1(model_xgb.predict(full_test_X))

print(error(test_Y, test_pred))
print(error_mse(test_Y, test_pred))


# ## MLP

# In[25]:


import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras import losses


# In[26]:


tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(42)
np.random.seed(42)


# In[27]:


mlp_feed = df_train[features]


# In[28]:


scaler = StandardScaler()
scaler.fit(mlp_feed)


# In[29]:


X_train = scaler.transform(mlp_feed)


# In[30]:


X_tr, X_val, y_tr, y_val = train_test_split(X_train, np.log(Y), random_state = 3, test_size=0.20)


# In[31]:


model = Sequential()
model.add(Dense(10, input_dim = X_train.shape[1],  activation="relu"))
model.add(Dense(1))

adam = optimizers.Adam()

model.compile(loss = losses.mean_squared_error, optimizer = adam)


# In[32]:


model.summary()


# In[33]:


monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True)  # save best model


# In[34]:


hist = model.fit(X_tr, y_tr, batch_size=1, validation_data = (X_val, y_val), callbacks=[monitor, checkpointer], verbose=1, epochs=100)


# In[35]:


model.load_weights('best_weights.hdf5')  # load weights from best model

# Measure accuracy
to_predict = scaler.transform(mlp_feed)
Ypred_mlp = np.exp(model.predict(to_predict))
print(error(df_train['SalePrice'].values,Ypred_mlp))
print(error_mse(df_train['SalePrice'].values, Ypred_mlp))


# #### test dataset

# In[36]:


full_test_X = df_test[features]
test_Y = df_test['SalePrice'].values

test_pred = np.exp(model.predict(full_test_X))

print(error(test_Y, test_pred))
print(error_mse(test_Y, test_pred))

