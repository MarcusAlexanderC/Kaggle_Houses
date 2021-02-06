import joblib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import seaborn as sns
from category_encoders.target_encoder import TargetEncoder
from lightgbm import LGBMRegressor
from sklearn.compose import make_column_transformer, ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer, mean_squared_log_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV 
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import plot_importance
from xgboost import XGBRegressor

 
train = pd.read_csv('kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('kaggle/input/house-prices-advanced-regression-techniques/test.csv')
sample_sub = pd.read_csv('kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

# Define functions.
def clean_drop_cols(df):
    df['Alley'] = df['Alley'].map(drop_dict_2lvls)
    df['PoolQC'] = df['PoolQC'].map(drop_dict_4lvls_PQC)
    df['Fence'] = df['Fence'].map(drop_dict_4lvls_F)
    df['FireplaceQu'] = df['FireplaceQu'].map(drop_dict_5lvls)
    return(df)

def clean_extr_ord_cols(df):
    for my_list, my_dict in ((ord_4lvls_cols, ord_dict_4lvls), (ord_5lvls_cols, ord_dict_5lvls), (ord_6lvls_cols, ord_dict_6lvls)):
        for col in my_list:
            df[col] = df[col].map(my_dict)
    return df

def clean_supp_extr_ord_cols(df):    
    df['GarageFinish'] = df['GarageFinish'].map(supp_ord_dict_3lvls_GF)
    df['PavedDrive'] = df['PavedDrive'].map(supp_ord_dict_3lvls_PD)
    df['Utilities'] = df['Utilities'].map(supp_ord_dict_4lvls)
    df['Functional'] = df['Functional'].map(supp_ord_dict_8lvls)
    return df

def select_cols_min_corr(cols, min_corr):
    abs_corrs = abs(pd.concat([X[cols], y], axis = 1).corr(method = 'spearman')['SalePrice'])
    cols_sel = list(abs_corrs.loc[abs_corrs > min_corr].index)
    cols_sel.remove('SalePrice')
    return cols_sel

def get_train_val_sets(X, y, cols):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)
    return X_train[cols], X_val[cols], y_train, y_val

def rmsle(y_true, y_pred): # Custom scoring function
    y_pred[y_pred < 0] = 0 # Replace negative predictions with 0.
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better = False)

def print_cv_val_score(my_s, print_best_est = True):    
    best_est = my_s.best_estimator_
    best_est.fit(X_train, y_train)
    y_pred = best_est.predict(X_val)
    val_score = rmsle(y_val, y_pred)
    best_CV_score = my_s.best_score_    
    print('Best CV score:', format(-best_CV_score, 'E'))
    print('Validation score:', format(val_score, 'E'))
    if print_best_est:
        print(best_est)

def get_sub_csv(my_s, cols, name_csv):
    best_est = my_s.best_estimator_
    #best_est.fit(X_train, y_train)
    best_est.fit(X[cols], y)
    X_test = test[cols]
    y_pred = best_est.predict(X_test)
    #y_pred = np.expm1(y_pred)
    test_submission = pd.DataFrame({'Id':test['Id'], 'SalePrice':y_pred})
    test_submission.to_csv(name_csv, index=False)
    
# Split train into features and target variable.
X = train.drop(['Id', 'SalePrice'], axis = 1)
y = train['SalePrice']

print('No. features:', X.shape[1])

# Drop columns with too many missing values (more than 10% missing).
max_fraction_null = 0.10
nrows = X.shape[0]
drop_cols = X.columns[X.isnull().sum() > nrows * max_fraction_null]
#X = X.drop(drop_cols, axis = 1) # We will keep drop_cols.
#test = test.drop(drop_cols, axis = 1)

# Clean drop_cols
drop_dict_2lvls = {'Grvl':0, 'Pave':1}
drop_dict_4lvls_PQC = {'Fa':0, 'TA':1, 'Gd':2, 'Ex':3}
drop_dict_4lvls_F = {'MnWw':0, 'GdWo':1, 'MnPrv':2, 'GdPrv':3}
drop_dict_5lvls = {'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}

X = clean_drop_cols(X)
test = clean_drop_cols(test)

# Set numeric and extracted ordinal features.
num_cols = list(X.select_dtypes(np.number).columns)
num_cols.remove('MSSubClass')

obj_cols = list(X.select_dtypes('O').columns)
temp = obj_cols.copy()
for element in ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', \
                'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', \
                'PavedDrive', 'SaleType', 'SaleCondition', 'MiscFeature']:
    temp.remove(element)
extr_ord_cols = temp

# Map the levels in extr_ord_cols to ordinal quantities.
ord_4lvls_cols = ['BsmtExposure']
ord_5lvls_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']
ord_6lvls_cols = ['BsmtFinType1', 'BsmtFinType2']

ord_dict_4lvls = {'No':0, 'Mn':1, 'Av':2, 'Gd':3}
ord_dict_5lvls = {'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}
ord_dict_6lvls = {'Unf':0, 'LwQ':1, 'Rec': 2, 'BLQ':3, 'ALQ':4, 'GLQ':5}

X = clean_extr_ord_cols(X)
test = clean_extr_ord_cols(test)

# Set supplementary manually extracted ordinal features.
supp_extr_ord_cols = ['Utilities', 'Functional', 'GarageFinish', 'PavedDrive']

supp_ord_dict_3lvls_GF = {'Unf':0, 'RFn':1, 'Fin':2}
supp_ord_dict_3lvls_PD = {'N':0, 'P':1, 'Y':2}
supp_ord_dict_4lvls = {'ELO':0, 'NoSeWa':1, 'NoSewr':2, 'AllPub':3}
supp_ord_dict_8lvls = {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2': 5, 'Min1':6, 'Typ':7}

X = clean_supp_extr_ord_cols(X)
test = clean_supp_extr_ord_cols(test)

# Restructure {num_cols + extr_ord_cols + supp_extr_ord_cols} into {num_cols + ord_cols}.
temp = num_cols.copy()
num_cols_keep = []
ord_cols = []

for element in ['LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
                'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']:
    num_cols_keep.append(element)
    temp.remove(element)

for element in ['OverallQual', 'OverallCond']:
    ord_cols.append(element)
    temp.remove(element)
    
# Set num_cols and ord_cols.
num_cols = num_cols_keep
ord_cols = ord_cols +  extr_ord_cols + supp_extr_ord_cols

# Add cols from drop_cols
num_cols = num_cols + ['LotFrontage']
ord_cols = ord_cols + ['Alley', 'Fence', 'FireplaceQu', 'PoolQC']

# Set the minimum correlation with the target variable.
#min_corr = 0.03125
#min_corr = 0.00000
    
# Apply minimum correlation filter.
#print('No. features before minimum correlation filter:', len(num_cols) + len(ord_cols))

#set_num_cols = set(num_cols)
#set_ord_cols = set(ord_cols)
#set_num_cols_f = set(select_cols_min_corr(num_cols, min_corr))
#set_ord_cols_f = set(select_cols_min_corr(ord_cols, min_corr))
#print('Features removed from num_cols:', set_num_cols.difference(set_num_cols_f))
#print('Features removed from ord_cols:', set_ord_cols.difference(set_ord_cols_f))

#num_cols = select_cols_min_corr(num_cols, min_corr)
#ord_cols = select_cols_min_corr(ord_cols, min_corr)

#print('No. features after minimum correlation filter:', len(num_cols) + len(ord_cols))

# Replace NAs.
#X['GarageYrBlt'] = X['GarageYrBlt'].fillna(-999)
#test['GarageYrBlt'] = test['GarageYrBlt'].fillna(-999)

#X[extr_ord_cols] = X[extr_ord_cols].fillna(-999)
#test[extr_ord_cols] = test[extr_ord_cols].fillna(-999)

#X[supp_extr_ord_cols] = X[supp_extr_ord_cols].fillna(-999)
#test[supp_extr_ord_cols] = test[supp_extr_ord_cols].fillna(-999)

# Take the logarithm of LotArea.
#X['LotArea'] = np.log(X['LotArea'])
#test['LotArea'] = np.log(test['LotArea'])

# Set the list of possible categorical features.
cat_cols = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition']
cat_cols = cat_cols + ['MiscFeature'] # Feature from drop_cols that could be considered categorical.

# Assert that there are no duplicates.
assert len(num_cols) == len(set(num_cols))
assert len(ord_cols) == len(set(ord_cols))
assert len(cat_cols) == len(set(cat_cols))

#Print the no. features in each category.
print('No. numerical features:', len(num_cols))
print('No. ordinal features:', len(ord_cols))
print('No. (possible) categorical features:', len(cat_cols), '\n')

# Set the list of categorical features to be used in the model.
#cat_cols = ['HasPool']
#cat_cols = ['CentralAir']
#cat_cols = ['Street']
#cat_cols = ['CentralAir', 'Street']
#cat_cols = ['MiscFeature']

# Print all the features considered
print('num_cols:', num_cols, '\n')
print('ord_cols:', ord_cols, '\n')
print('cat_cols:', cat_cols, '\n')

X_train, X_val, y_train, y_val = get_train_val_sets(X, y, num_cols + ord_cols + cat_cols)   
