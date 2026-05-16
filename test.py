from joblib import load
from SupremeCourtTimeMachine import standardize_term, determine_era
from demo import make_predictions, compare_predictions
from oyez_script import get_case_facts
import pandas as pd
import numpy as np

test_df = pd.read_csv('./data/fully_processed_data_post_2021_inc')
test_df.dropna(inplace=True)
test_df = test_df.iloc[:, :10]

# Tests for SupremeCourtTimeMachine notebook
def test_standardize_term():
    assert standardize_term('1973') == 1973
    assert standardize_term('1973-001-001') == 1973

def test_determine_era():
    assert determine_era(0) == 'unk'
    assert determine_era('1971') == 'pre-rehnquist'
    assert determine_era(1941) == 'unk'
    assert determine_era(1942) == 'pre-rehnquist'
    assert determine_era(1985) == 'pre-rehnquist'
    assert determine_era(1989) == 'post-rehnquist_inc'
    assert determine_era(2024) == 'post-rehnquist_inc'

def test_SupremeCourtTimeMachine():
    test_standardize_term()
    test_determine_era()

# Tests for demo.py
test_model = load('./saved_models/post_rehnquist_inc_model')
predictions = make_predictions(test_df, test_model)[:10]
actual_predictions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
inverse_predictions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

def test_make_predictions():
    assert np.array_equal(np.array(predictions), np.array(actual_predictions))

def test_compare_predictions():
    assert compare_predictions(actual_predictions, actual_predictions) == 1.0
    assert compare_predictions(actual_predictions, np.ones(10)) == 0.9
    assert compare_predictions(actual_predictions, inverse_predictions) == 0.0

def test_demo():
    test_make_predictions()
    test_compare_predictions()

# Tests for oyez_script.py
def test_get_case_facts():
    row = {
        'docket': '19-1392', 
        'term': 2021, 
        'facts': '',
        'first_party_winner': '',
        'era': '',
    }
    actual_facts_first_40 = 'In 2018, Mississippi passed a law called'    
    assert get_case_facts(row)[:40] == actual_facts_first_40

def test_oyez_script():
   test_get_case_facts()

if __name__ == '__main__':
    print('-----STARTING TESTS-----')
    test_SupremeCourtTimeMachine()
    test_demo()
    test_oyez_script()

    print('Tests passed')
    print('-----ENDING TESTS-----')