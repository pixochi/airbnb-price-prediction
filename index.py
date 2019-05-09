import pandas as pd
import numpy as np
from datetime import date

binary_string_to_number_converter = lambda x: (
    1 if x == 't' else (
        0 if x == 'f' else x # don't convert x if it's NaN
    )
)

# Importing the dataset
dataset = pd.read_csv(
        'airbnb_cph_listings.csv',
        # These dataset columns have mixed data types
        dtype = {
            'zipcode': str,
            'weekly_price': str,
            'monthly_price': str
        },
        parse_dates = [
               'host_since',
               'first_review',
               'last_review'
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

# Preprocessing
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
]

X_features = dataset.loc[:, columns_to_keep]
y_label = dataset.loc[:, ['price']]
y_label = y_label['price'].str.replace('\$|,', '').astype(float)

# Only very few rows don't have 'host_since' registered(9)
dataset['host_since'].dropna(inplace = True)

today = date.today()
days_since = pd.DataFrame({
#    not using days_since_last_review and days_since_first_review
#    because thousands of rows were missing data
    'days_since_host_registration': today - dataset['host_since'],
}).applymap(
    lambda x: x.days # get days as number
).apply(
    # use mean of each column for NaN values
    lambda x: x.fillna(x.mean()),
    axis = 1
)

# join X_features with a new feature
X_features = pd.DataFrame(
    pd.concat(
        [X_features, days_since],
        axis = 1
    )
)
    
# drop rows where any feature value is NaN (host_is_superhost, beds, bedrooms, bathrooms)
# cca only 15 rows for the features mentioned above
X_features = X_features.dropna()

# convert price for extra_people into float
X_features['extra_people'] = X_features['extra_people'].str.replace('\$|,', '').astype(float)

# get all types of host_verifications as new binary features
from ast import literal_eval

all_host_verifications_types = X_features['host_verifications'].map(
    lambda x: literal_eval(x) # read stringified list as a normal list
).sum()
all_host_verifications_types = np.unique(all_host_verifications_types)

for verifications_type in all_host_verifications_types:
    X_features[verifications_type] = X_features['host_verifications'].str.contains(verifications_type, regex = False)

X_features.drop(['host_verifications'], axis=1, inplace = True)
    
# TODO: get all amenities as new binary features
#all_amenities = X_features['amenities'].map(
#    lambda x: literal_eval(x.replace('{', '[').replace('}', ']')) # read stringified list as a normal list
#).sum()
#all_amenities = np.unique(all_amenities)
#
#
#for amenity in all_amenities:
#    X_features[amenity] = X_features['amenities'].str.contains(amenity, regex = False)
#
# X_features.drop('amenities', inplace = True)
