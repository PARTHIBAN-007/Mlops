import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import Cleaner


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'SalesChannelID': [1, 1, 1, 1],
        'VehicleAge': [1, 1, 1, 1],
        'DaysSinceCreated': [1, 1, 1, 1],
        'AnnualPremium': ['£1,200', '£2,500', '£3,000', '£5,000'],
        'Gender': ['Male', np.nan, 'Female', 'Male'],
        'RegionID': [np.nan, 2, 3, np.nan],
        'Age': [30, np.nan, 25, 40],
        'HasDrivingLicense': [1, np.nan, 1, np.nan],
        'Switch': [0, 1, np.nan, 0],
        'PastAccident': [np.nan, 'Yes', 'No', 'Yes']
    })

@pytest.fixture
def cleaner():
    return cleaner()


def test_clean_data(cleaner , sample_data):
    cleaned_data = cleaner.clean_data(sample_data.copy())

    assert 'id' not in cleaned_data.columns
    assert 'SalesChannelID' not in cleaned_data.columns
    assert 'VehicleAge' not in cleaned_data.columns
    assert 'DaysSinceCreated' not in cleaned_data.columns

    assert cleaned_data['AnnualPremium'].dtype ==float

    assert not cleaned_data['Gender'].isnull().any()
    assert not cleaned_data['RegionID'].isnull().any()


    assert not cleaned_data['Age'].isnull().any()
    assert not cleaned_data['HasDrivingLicense'].isnull().any()
    assert (cleaned_data['HasDrivingLicense']==1).all()

    assert not cleaned_data['switch'].isnull().any()
    assert (cleaned_data['switch']==-1).any()

    assert not cleaned_data['PastAccident'].isnull().any()
    assert (cleaned_data['PastAccident']=='Unknown').any()

    Q1 = cleaned_data['AnnualPremium'].quantile(0.25)
    Q3 = cleaned_data['AnnualPremium'].quantile(0.75)

    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5*IQR
    assert (cleaned_data['AnnualPremium']<=upper_bound).all() 




