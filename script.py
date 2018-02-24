#AUTHORS: EMANUELE ALESSI - IVAN FERRANTE
#For more informations check the IPYNB/PDF files
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor

def add_dummies(dataset, column, variable):
    tmp = pd.get_dummies(dataset[column])
    tmp = tmp.drop(variable, axis = 1)
    for col in tmp.columns:
        dataset[column + '_' + col] = tmp[col]
    return dataset.drop(column, axis = 1)

def load_dataset():
    print('Loading the dataset...')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def feature_engineering(train, test):
    print('Cleaning the dataset...')
    t = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 700000)].index)
    train_salePrice = t['SalePrice']
    train_shape = t.shape
    dataset = pd.concat((t, test)).reset_index(drop=True)
    dataset['PoolQC'] = dataset['PoolQC'].fillna(0)
    d = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    dataset['PoolQC'] = dataset['PoolQC'].replace(d)
    dataset['MiscFeature'] = dataset['MiscFeature'].fillna('NA')
    dataset = add_dummies(dataset, 'MiscFeature', 'NA')
    dataset['Alley'] = dataset['Alley'].fillna('NA')
    dataset = add_dummies(dataset, 'Alley', 'NA')
    dataset['Fence'] = dataset['Fence'].fillna('NA')
    dataset = add_dummies(dataset, 'Fence', 'NA')
    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(0)
    d = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataset['FireplaceQu'] = dataset['FireplaceQu'].replace(d)
    dataset['LotFrontage'] = dataset.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    dataset['GarageQual'] = dataset['GarageQual'].fillna(0)
    dataset['GarageCond'] = dataset['GarageCond'].fillna(0)
    d = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataset['GarageQual'] = dataset['GarageQual'].replace(d)
    dataset['GarageCond'] = dataset['GarageCond'].replace(d)
    dataset['GarageFinish'] = dataset['GarageFinish'].fillna(0)
    d = {'Unf': 1, 'RFn': 2, 'Fin': 3}
    dataset['GarageFinish'] = dataset['GarageFinish'].replace(d)
    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)
    dataset['GarageType'] = dataset['GarageType'].fillna('NA')
    dataset = add_dummies(dataset, 'GarageType', 'NA')
    dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(0)
    d = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    dataset['BsmtExposure'] = dataset['BsmtExposure'].replace(d)
    dataset['BsmtCond'] = dataset['BsmtCond'].fillna(0)
    dataset['BsmtQual'] = dataset['BsmtQual'].fillna(0)
    d = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataset['BsmtCond'] = dataset['BsmtCond'].replace(d)
    dataset['BsmtQual'] = dataset['BsmtQual'].replace(d)
    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(0)
    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(0)
    d = {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].replace(d)
    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].replace(d)
    dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')
    dataset = add_dummies(dataset, 'MasVnrType', 'None')
    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
    dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].value_counts().index[0])
    dataset = add_dummies(dataset, 'MSZoning', 'C (all)')
    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0)
    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(0)
    dataset = dataset.drop('Utilities', axis=1)
    dataset['Functional'] = dataset['Functional'].fillna(dataset['Functional'].value_counts().index[0])
    dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].value_counts().index[0])
    dataset = add_dummies(dataset, 'Functional', 'Sev')
    dataset = add_dummies(dataset, 'Electrical', 'Mix')
    dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(0)
    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].value_counts().index[0])
    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].value_counts().index[0])
    dataset = add_dummies(dataset, 'Exterior1st', 'AsbShng')
    dataset = add_dummies(dataset, 'Exterior2nd', 'AsphShn')
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(0)
    dataset['GarageCars'] = dataset['GarageCars'].fillna(0)
    dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(0)
    dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(0)
    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].value_counts().index[0])
    d = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    dataset['KitchenQual'] = dataset['KitchenQual'].replace(d)
    dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].value_counts().index[0])
    dataset = add_dummies(dataset, 'SaleType', 'Oth')
    dataset['GarageArea'] = dataset['GarageArea'].fillna(0)
    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
    for column in dataset.select_dtypes(exclude=[np.number]).columns:
        dataset = add_dummies(dataset, column, dataset[column].value_counts().index[-1])
    dataset = dataset.drop(['Id', 'SalePrice'], axis=1)
    tr = dataset[: train_shape[0]]
    te = dataset[train_shape[0]:]
    return tr, te, train_salePrice

def make_predictions(train, test, train_salePrice):
    alphas = [27.6]
    n_estimators = 395
    X = train
    y = np.log1p(train_salePrice.values)
    print('Predicting house prices with:')
    print('RidgeCV(alphas = ' + str(alphas) + ')')
    print('GradientBoostingRegressor(n_estimators = ' + str(n_estimators) + ')')
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv_predictions = ridge_cv.fit(X, y).predict(test)
    gbm = GradientBoostingRegressor(n_estimators=n_estimators)
    gbm_predictions = gbm.fit(X, y).predict(test)
    predictions = np.expm1(ridge_cv_predictions) * 0.7 + np.expm1(gbm_predictions) * 0.3
    print('Saving predictions in submission.csv')
    submission = pd.DataFrame()
    submission['Id'] = test_id
    submission['SalePrice'] = predictions
    submission.to_csv('submission.csv', index=False)
    print('Done!')


train, test = load_dataset()
train_id = train['Id']
test_id = test['Id']
train, test, train_salePrice = feature_engineering(train, test)
make_predictions(train, test, train_salePrice)