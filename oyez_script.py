import pandas as pd
import oyez_api_wrapper as oyez

# Fetch case facts from oyez API given a term and a docket
def get_case_facts(row):
    try:
        return oyez.court_case(row['term'], row['docket']).get_case_facts()
    except Exception as e:
        return

if __name__ == '__main__':
    # Read SCOTUS data
    cases_df = pd.read_csv('./data/SCDB_2023_01_caseCentered_Docket.csv', encoding='ISO-8859-1')

    # Filter SCOTUS Database for relevant/useful columns
    cases_selected_columns_df = cases_df[['caseId', 'docketId', 'docket', 'term', 'caseName', 'issueArea', 'partyWinning']]

    # Fetch facts from and put them in the data
    cases_selected_columns_df['facts'] = cases_selected_columns_df.apply(get_case_facts, axis=1)

    # Filter dataframe for opinions with a facts paragraph
    df = cases_selected_columns_df[cases_selected_columns_df['facts'].notna()]

    print(df.head())

    # Save updated SCOTUS data to avoid running the above again
    df.to_csv('./data/SCOTUS_data_with_facts')

    # Check to make sure it looks good
    df_from_file = pd.read_csv('./data/SCOTUS_data_with_facts')
    print(df_from_file.head())