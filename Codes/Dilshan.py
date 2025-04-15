import featuretools as ft
from featuretools.selection import (
    remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features,
)
import numpy as np
import pandas as pd

class DilFeat:
    def __init__(self):
        # List all available transformation primitives in Featuretools
        all_trans_primitives = list(ft.primitives.get_transform_primitives().keys())

        # Exclude the 'Equal' and 'Not Equal' primitive from the list
        self.filtered_trans_primitives = [primitive for primitive in all_trans_primitives if primitive != 'equal' and primitive !='not_equal']

    def apply_feature_tools(self, X_train, X_test):
        # Create an EntitySet for the training data
        es_train = ft.EntitySet(id='Training_set')
        es_train = es_train.add_dataframe(
            dataframe_name='data',
            dataframe=X_train,
            index='index'  # Ensure this is a unique index
        )

        # Generate and save features on the training set
        feature_matrix_train, feature_defs = ft.dfs(
            entityset=es_train,
            target_dataframe_name='data',
            trans_primitives=self.filtered_trans_primitives
        )

        # Create an EntitySet for the testing data
        es_test = ft.EntitySet(id='Testing_set')
        es_test = es_test.add_dataframe(
            dataframe_name='data',
            dataframe=X_test,
            index='index'  # Ensure this is a unique index
        )

        # Apply the same transformations to the test set
        feature_matrix_test = ft.calculate_feature_matrix(
            features=feature_defs,
            entityset=es_test
        )

        categorical_cols = feature_matrix_train.select_dtypes(include=['object', 'category']).columns.tolist()        

        return feature_matrix_train, feature_matrix_test, feature_defs, categorical_cols

    def encode_categorical_features(self, X_train, X_test, categorical_cols, feature_defs):
        # Encoding the categorical features for the training set
        fm_encoded_train, f_encoded_train = ft.encode_features(X_train, features=feature_defs, to_encode=categorical_cols, inplace=True)

        # Encoding the categorical features for the testing set
        fm_encoded_test, f_encoded_test = ft.encode_features(X_test, features=feature_defs, to_encode=categorical_cols, inplace=True)

        # Find columns in training not in testing and add them with default value 0
        train_not_test = fm_encoded_train.columns.difference(fm_encoded_test.columns)
        for col in train_not_test:
            fm_encoded_test[col] = 0

        # Find columns in testing not in training and drop them
        test_not_train = fm_encoded_test.columns.difference(fm_encoded_train.columns)
        fm_encoded_test = fm_encoded_test.drop(columns=test_not_train)

        return fm_encoded_train, fm_encoded_test, f_encoded_train

    def feature_selection(self, X_train, X_test, feature_defs):
        # Removing features in the train set that have only a single unique value
        X_train_filtered, new_features = remove_single_value_features(X_train, features=feature_defs)

        # Removing features in the train set that are highly correlated
        X_train_filtered, new_features = remove_highly_correlated_features(X_train_filtered, features=new_features)

        # Dropping any columns in the train set that contain NaN values
        X_train_filtered = X_train_filtered.dropna(axis=1, how='any')

        # Selecting the same subset of features from the test set
        X_test_filtered = X_test[X_train_filtered.columns]

        return X_train_filtered, X_test_filtered

    def process_data(self, X_train, X_test):
        # Combine the training and testing datasets with keys to create a MultiIndex
        combined = pd.concat([X_train, X_test], keys=['train', 'test'])
        
        # Replace infinite values with NaN
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop columns with any NaN values
        combined.dropna(axis=1, how='any', inplace=True)
        
        # Separate the combined dataset back into training and testing sets using the keys
        X_train_processed = combined.xs('train')
        X_test_processed = combined.xs('test')
        
        return X_train_processed, X_test_processed
    

    def transform(self, X_train, X_test):
        # Apply feature engineering
        X_train_transformed, X_test_transformed, feature_defs, categorical_cols = self.apply_feature_tools(X_train, X_test)

        # Encode categorical features
        X_train_encoded, X_test_encoded, X_train_encoded_defs = self.encode_categorical_features(
            X_train_transformed, X_test_transformed, categorical_cols = categorical_cols, feature_defs = feature_defs)

        # Perform feature selection
        X_train_selected, X_test_selected = self.feature_selection(X_train_encoded, X_test_encoded, X_train_encoded_defs)

        # Removing nan columns
        X_train_finalized, X_test_finalized = self.process_data(X_train_selected, X_test_selected)

        return X_train_finalized, X_test_finalized
