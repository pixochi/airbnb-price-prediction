import pandas as pd
#import numpy as np
from datetime import datetime

# =============================================================================
# 1. IMPORTING THE DATASET
# =============================================================================

binary_string_to_number_converter = lambda x: (
    1 if x == 't' else (
        0 if x == 'f' else -1 # convert x to -1 if it's NaN
    )
)

dataset = pd.read_csv(
        'airbnb_cph_listings.csv',
        # These dataset columns have mixed data types
        dtype = {
            'zipcode': str,
            'weekly_price': str,
            'monthly_price': str,
        },
        parse_dates = [
               'host_since',
               'first_review',
               'last_review',
        ],
        converters = {
            'host_is_superhost': binary_string_to_number_converter,
            'host_has_profile_pic': binary_string_to_number_converter,
            'host_identity_verified': binary_string_to_number_converter,
            'is_location_exact': binary_string_to_number_converter,
            'has_availability': binary_string_to_number_converter,
            'requires_license': binary_string_to_number_converter, 
            'instant_bookable': binary_string_to_number_converter,
            'is_business_travel_ready': binary_string_to_number_converter,
            'require_guest_profile_picture': binary_string_to_number_converter,
            'require_guest_phone_verification': binary_string_to_number_converter,
        }
)

# =============================================================================
# 2. PREPROCESSING
# =============================================================================

columns_to_keep = [
    'host_is_superhost',
    'host_listings_count',
    'host_verifications',
    'host_has_profile_pic',
    'host_identity_verified',
    'neighbourhood_cleansed',
    'is_location_exact',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'bed_type',
    'room_type',
    'property_type',
    'amenities',
    'guests_included',
    'extra_people',
    'minimum_nights',
    'maximum_nights',
    'has_availability',
    'availability_30', 
    'availability_60',
    'availability_90',
    'availability_365',
    'number_of_reviews',
    'number_of_reviews_ltm',
    'requires_license', 
    'instant_bookable',
    'is_business_travel_ready',
    'cancellation_policy',
    'require_guest_profile_picture',
    'require_guest_phone_verification',
    'host_since',
    'price',
]

dataset_processed = dataset.loc[:, columns_to_keep]

# Only very few rows don't have 'host_since' registered(9)
dataset_processed['host_since'].dropna(inplace = True)

today = datetime.today()
days_since = pd.DataFrame({
#    not using days_since_last_review and days_since_first_review
#    because thousands of rows were missing data
    'days_since_host_registration': today - dataset_processed['host_since'],
}).applymap(
    lambda x: x.days # get days as number
).apply(
    # use mean of each column for NaN values
    lambda x: x.fillna(x.mean()),
    axis = 1    
)

# join X_features with a new feature
dataset_processed = pd.DataFrame(
    pd.concat(
        [dataset_processed, days_since],
        axis = 1
    )
)
dataset_processed.drop(['host_since'], axis = 1)
    
# drop rows where feature values are NaN
# cca only 15 rows for each feature
dataset_processed = dataset_processed.dropna(subset = ['beds', 'bedrooms', 'bathrooms'])
# NaN has been converted to -1 in dataset import
dataset_processed = dataset_processed[dataset_processed['host_is_superhost'] != -1]

# convert price for extra_people into float
dataset_processed['extra_people'] = dataset_processed['extra_people'].str.replace('\$|,', '').astype(float)

# get all types of host_verifications as new binary features
#from ast import literal_eval
#
#all_host_verifications_types = dataset_processed['host_verifications'].map(
#    lambda x: [] if x == 'None' else literal_eval(x) # read stringified list as a normal list
#).sum()
#all_host_verifications_types = np.unique(all_host_verifications_types)
#
#for verifications_type in all_host_verifications_types:
#    dataset_processed[verifications_type] = dataset_processed['host_verifications'].str.contains(verifications_type, regex = False) * 1 # multiplying boolean by 1 converts bool to int


dataset_processed['host_verifications_count'] = dataset_processed['host_verifications'].str.count("'\w+'")
dataset_processed.drop(['host_verifications'], axis = 1, inplace = True)

#import re
## get all amenities as new binary features
#all_amenities = dataset_processed['amenities'].map(
#    lambda x: re.sub(
#                '{(\w+)',
#                r'{"\1"',
#                re.sub(
#                    ',(\w+)', # add missing quotation marks around each amenity
#                    r',"\1"',
#                    x
#                )
#            ).replace(
#                '{', '['
#            ).replace(
#                '}', ']'
#            ) 
#).map(lambda x: re.findall(r'"\s*([^"]*?)\s*"', x)).sum() # convert a stringified list to a normal list
#
#all_amenities = np.unique(all_amenities)
#
#for amenity in all_amenities:
#    dataset_processed[amenity] = dataset_processed['amenities'].str.contains(amenity, regex = False) * 1 # multiplying boolean by 1 converts bool to int
#
#dataset_processed['amenities'] = dataset_processed['amenities'].map(
#    lambda x: re.sub(
#                '{(\w+)',
#                r'{"\1"',
#                re.sub(
#                    ',(\w+)', # add missing quotation marks around each amenity
#                    r',"\1"',
#                    x
#                )
#            )
#)
                
dataset_processed['amenities_count'] = dataset_processed['amenities'].str.count('[a-zA-Z0-9_ \/-]+')
dataset_processed.drop('amenities', axis=1, inplace = True)

# =============================================================================
# 3. ENCODING CATEGORICAL FEATURES
# =============================================================================

from sklearn.preprocessing import LabelEncoder
labelEncoderX = LabelEncoder()
dataset_processed['neighbourhood_cleansed'] = labelEncoderX.fit_transform(dataset_processed['neighbourhood_cleansed'])
dataset_processed['bed_type'] = labelEncoderX.fit_transform(dataset_processed['bed_type'])
dataset_processed['cancellation_policy'] = labelEncoderX.fit_transform(dataset_processed['cancellation_policy'])
dataset_processed['room_type'] = labelEncoderX.fit_transform(dataset_processed['room_type'])
dataset_processed['property_type'] = labelEncoderX.fit_transform(dataset_processed['property_type'])

categorical_features = [
    'neighbourhood_cleansed',
    'bed_type',
    'cancellation_policy',
    'room_type',
    'property_type',
]

X_features = dataset_processed.drop(['price', 'host_since'], axis = 1)
y_label = dataset_processed.loc[:, ['price']]
y_label = y_label['price'].str.replace('\$|,', '').astype(float)

# https://stackoverflow.com/questions/51748260/python-onehotencoder-using-many-dummy-variables-or-better-practice
# 'drop_first=True' saves you from the dummy variable trap
X_features = pd.get_dummies(X_features,columns = categorical_features, drop_first = True)


# =============================================================================
# 4. SPLITTING THE DATASET INTO TRAINING/TESTING SETS
# =============================================================================
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X_features, y_label, random_state = 0, test_size = 0.2)

# =============================================================================
# 5. FEATURE SCALING
# =============================================================================

from sklearn.preprocessing import StandardScaler

# why to fit the scaler on training data only
# https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.transform (XTest)

# =============================================================================
# 6. LINEAR REGRESSION MODEL
# =============================================================================

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(XTrain, yTrain)

# Predictions
yPred = regressor.predict(XTest)

# =============================================================================
# 6.1 LINEAR REGRESSION MODEL - PREDICTIONS - BEFORE OPTIMIZATION
# =============================================================================
from sklearn.metrics import mean_squared_error, r2_score
print("\nBEFORE OPTIMIZATION")
print("Mean squared error: %.2f"
      % mean_squared_error(yTest, yPred))
# variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(yTest, yPred))


# =============================================================================
# 6.1 LINEAR REGRESSION MODEL - OPTIMIZING - BACKWARD ELIMINATION
# =============================================================================

import numpy as np
import statsmodels.formula.api as sm
from sklearn.metrics import mean_squared_error, r2_score

def backwardEliminateFeatures(features, y_label, p_threshold):    
    regressorOLS = sm.OLS(y_label, features).fit()
    # lower the pValue, higher the statistical significance
    pvalues = regressorOLS.pvalues
    
    max_pvalue_index = pvalues.argmax()
    
    if (max_pvalue_index == 'const'):
        max_pvalue_index = 0
    elif(max_pvalue_index.find('x') > -1):
        max_pvalue_index = int(max_pvalue_index[1:])

    max_pvalue = pvalues[max_pvalue_index]
    
    if(max_pvalue < p_threshold):
        return features
    else:
         features = np.delete(features, max_pvalue_index, axis=1)
         return backwardEliminateFeatures(features, y_label, p_threshold)

    
# add a column with the constant
X_features = np.append(np.ones((X_features.shape[0], 1)).astype(int), X_features, axis = 1)
X_features = backwardEliminateFeatures(X_features, y_label, 0.0001)

# =============================================================================
# 6.2 LINEAR REGRESSION MODEL - PREDICTIONS - AFTER OPTIMIZATION
# =============================================================================
XTrain, XTest, yTrain, yTest = train_test_split(X_features, y_label, random_state = 0, test_size = 0.2)

regressor = LinearRegression()
regressor.fit(XTrain, yTrain)

yPred = regressor.predict(XTest)

print("\nAFTER OPTIMIZATION")
print("Mean squared error: %.2f"
      % mean_squared_error(yTest, yPred))
# variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(yTest, yPred))










