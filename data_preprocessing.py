import pandas as pd
import numpy as np
from sklearn import preprocessing

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
    'marital_status', 'occupation', 'relationship', 'race', 
    'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
    'native_country', 'income_bracket'
]

CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital_status', 'occupation', 
    'relationship', 'race', 'sex', 'native_country'
]

CONTINUOUS_COLUMNS = [
    'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'
]

class data_loader:
    def __init__(self, data_path, skip_rows=0):
        self.data_path = data_path
        self.skip_rows = skip_rows

    def data_preprocessing(self):
        x_data = pd.read_csv(self.data_path, names=COLUMNS, skiprows=self.skip_rows)
        y_data = x_data['income_bracket'].apply(lambda x: '>50K' in x)    # train label: 1 if '>50K' else 0
        y_data = y_data.astype(int)     # boolean to int
        x_data.pop('income_bracket')        # remove 'income_bracket'
        for i in CATEGORICAL_COLUMNS:
            le = preprocessing.LabelEncoder()
            x_data[i] = le.fit_transform(x_data[i])
        x_data_categ = np.array(x_data[CATEGORICAL_COLUMNS])
        x_data_conti = np.array(x_data[CONTINUOUS_COLUMNS], dtype='float')
        scaler = preprocessing.StandardScaler()
        x_data_conti = scaler.fit_transform(x_data_conti)     # Standardize data
        
        return x_data, y_data, x_data_categ, x_data_conti


if  __name__ == '__main__':
    TRAIN_PATH = './dataset/adult.data'
    TEST_PATH = './dataset/adult.test'

    train_data = data_loader(TRAIN_PATH)
    x_train, y_train, x_train_categ, x_train_conti = train_data.data_preprocessing()
    print(x_train)
    
    test_data = data_loader(TEST_PATH, 1)
    x_test, y_test, x_test_categ, x_test_conti = test_data.data_preprocessing()
    print(x_test)
