import pandas as pd
from joblib import load

def make_predictions(df, model):
    return model.predict(df['facts'])

def compare_predictions(predictions_0, predictions_1):
    same_outcome_count = 0
    for i in range(len(predictions_0)):
        if predictions_0[i] == predictions_1[i]:
            same_outcome_count += 1
    
    return same_outcome_count / len(predictions_0)

def main():
    # Read in post-2021 data
    df = pd.read_csv('./data/fully_processed_data_post_2021_inc')
    
    # Load pre-trained models
    pre_rehnquist_model = load('./saved_models/pre_rehnquist_model')
    post_rehnquist_model = load('./saved_models/post_rehnquist_inc_model')

    # Predicts the outcome of a single case given a year to determine era and a facts paragraph describing the case
    def predict_case_outcome(year, facts):
        if not year or year < 1941:
            return 'Year too early, try again...'
        else:
            # determine model
            if 1941 <= year < 1986:
                result = pre_rehnquist_model.predict([facts,])
                print(result)
            elif 1986 <= year:
                result = post_rehnquist_model.predict([facts,])

            # determine result
            if result:
                return 'First party wins'
            else:
                return 'Second party wins'
            
    # Compute model prediction similaries
    prediction_similarity = compare_predictions(make_predictions(df, pre_rehnquist_model), 
                                                make_predictions(df, post_rehnquist_model))

    print('The pre-rehnquist-court and post-rehnquist-court (inclusive) predicted the same case outcome {:.2%}'.format(prediction_similarity))

    # UI Loop for single case outputs
    while True:
        # Input year
        print('Type \"exit\" to quit at any step:')
        year_input = input('Enter a year: ')
        if year_input.lower() == 'exit':
            print('...exited')
            break

        # Ensure year is a number
        try:
            year = int(year_input)
        except ValueError:
            print('Invalid year, try again: ')
            continue

        # Input facts
        facts = input('Enter a facts summary: ')
        if facts.lower() == 'exit':
            print('...exited')
            break

        # Call the predict function and display the result
        result = predict_case_outcome(year, facts)
        print(result)


if __name__ == "__main__":
    main()