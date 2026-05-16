# SupremeCourtTimeMachine
Use machine learning to quantify drift in SCOTUS, and predict conclusions of new cases.

oyez_script.py: (run from terminal)
    If fully_processd_data is not in the data folder, run this before doing anything. 
    It will take between 1 and 2 hours, and will generate the SCOTUS_data_with_facts file (assuming you also have the SCDB_2023_01_caseCentered_Docket file)
    * See appendix of report for dependencies

SupremeCourtTimeMachine.ipynb: (jupyter notebook)
    This is the notebook behind almost all the data processing, NLP, and model training. 
    It should be viewed as the main file of the project.
    If you wish to run the demo, this shouldn't be nessesary, as there is already an executable version of it saved.
    * See appendix of report for dependencies

demo.py (run from terminal)
    This will load pre-trained models and make and compare their predicitons on post-2021 case data.
    It will also display the single-case UI, where you can enter a year between 1941 and 2024, 
    along with a fact summary, and get a case outcome prediction.
    * See appendix of report for dependencies

test.py (run from terminal)
    * Python files must execute before modules or functions are exported. This applies to jupyter notebooks.
    Unfortunately, that means the test file must run SupremeCourtTimeMachine. 
    There is a tests starting and tests ending indicator message to clarify this.
    * See appendix of report for dependencies


