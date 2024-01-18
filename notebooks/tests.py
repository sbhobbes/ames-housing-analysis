import my_module
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from loguniform_int import loguniform_int


def test_invocation():
    features, target = my_module.get_features_and_target(
        csv_file='../data/adult-census.csv',
        target_col='class'
    )


@pytest.mark.parametrize(
    'csv,target',
    [
        ('../data/adult-census.csv', 'class'),
        ('../data/ames.csv', 'Sale_Price')
    ]
)
def test_return_types(csv, target):
    features, target = my_module.get_features_and_target(
        csv_file=csv,
        target_col=target
    )
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)


def test_cols_make_sense():
    features, target = my_module.get_features_and_target(
        csv_file='../data/adult-census.csv',
        target_col='class'
    )
    # Load the data ourselves so we can double-check the columns
    df = pd.read_csv('../data/adult-census.csv')
    assert target.name in df.columns
    # Use a list comprehension to check all the feature columns
    assert all([feature_col in df.columns for feature_col in features])


@pytest.mark.parametrize(
    'csv', [ ['a', 'b', 'c'], 123 ]
)
def test_bad_input_error(csv):
    with pytest.raises(ValueError):
        features, target = my_module.get_features_and_target(
            csv_file=csv,
            target_col='Sale_Price'
        )


def test_make_preprocessor():
    features, target = my_module.get_features_and_target(
        csv_file='../data/adult-census.csv',
        target_col='class'
    )
    preprocessor = my_module.make_preprocessor(features)
    
    assert isinstance(preprocessor, ColumnTransformer), \
        f'Expected an object of the sklearn.compose.ColumnTransformer class, but got an object of type "{type(preprocessor)}" instead.'


@pytest.mark.parametrize('a, b', [(2, 30), (1, 100)])
def test_log_uniform_int(a, b):

    lu = loguniform_int(a, b)
    assert lu._distribution.args == (a, b)
