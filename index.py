import pandas as pd
import numpy as np
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
from ast import literal_eval

all_host_verifications_types = dataset_processed['host_verifications'].map(
    lambda x: [] if x == 'None' else literal_eval(x) # read stringified list as a normal list
).sum()
all_host_verifications_types = np.unique(all_host_verifications_types)

for verifications_type in all_host_verifications_types:
    dataset_processed[verifications_type] = dataset_processed['host_verifications'].str.contains(verifications_type, regex = False) * 1 # multiplying boolean by 1 converts bool to int

dataset_processed.drop(['host_verifications'], axis=1, inplace = True)

import re
# get all amenities as new binary features
all_amenities = dataset_processed['amenities'].map(
    lambda x: re.sub(
                '{(\w+)',
                r'{"\1"',
                re.sub(
                    ',(\w+)', # add missing quotation marks around each amenity
                    r',"\1"',
                    x
                )
            ).replace(
                '{', '['
            ).replace(
                '}', ']'
            ) 
).map(lambda x: re.findall(r'"\s*([^"]*?)\s*"', x)).sum() # convert a stringified list to a normal list

all_amenities = np.unique(all_amenities)

for amenity in all_amenities:
    dataset_processed[amenity] = dataset_processed['amenities'].str.contains(amenity, regex = False) * 1 # multiplying boolean by 1 converts bool to int

dataset_processed.drop('amenities', axis=1, inplace = True)

# =============================================================================
# 3. ENCODING CATEGORICAL FEATURES
# =============================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
dataset_processed['neighbourhood_cleansed'] = labelEncoderX.fit_transform(dataset_processed['neighbourhood_cleansed'])
dataset_processed['bed_type'] = labelEncoderX.fit_transform(dataset_processed['bed_type'])
dataset_processed['cancellation_policy'] = labelEncoderX.fit_transform(dataset_processed['cancellation_policy'])
dataset_processed['room_type'] = labelEncoderX.fit_transform(dataset_processed['room_type'])
dataset_processed['property_type'] = labelEncoderX.fit_transform(dataset_processed['property_type'])

categorical_features_indexes = [
    dataset_processed.columns.get_loc('neighbourhood_cleansed'),
    dataset_processed.columns.get_loc('bed_type'),
    dataset_processed.columns.get_loc('cancellation_policy'),
    dataset_processed.columns.get_loc('room_type'),
    dataset_processed.columns.get_loc('property_type'),
]

X_features = dataset_processed.drop(['price', 'host_since'], axis = 1)
y_label = dataset_processed.loc[:, ['price']]
y_label = y_label['price'].str.replace('\$|,', '').astype(float)

oneHotEncoder = OneHotEncoder(categorical_features = categorical_features_indexes)
X_features = oneHotEncoder.fit_transform(X_features).toarray()

# =============================================================================
# 4. FEATURE SCALING
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)

# =============================================================================
# 5. LINEAR REGRESSION MODEL
# =============================================================================

















