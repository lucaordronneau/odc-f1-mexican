import pandas as pd 

class ResultDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        result_data = pd.read_csv(self.filepath)
        return self.clean(result_data)

    def clean(self, df):
        excluded_gp = ['70th Anniversary Grand Prix', 'Pre-Season Test 2', 'Pre-Season Test 1', 
                       'Pre-Season Track Session', 'Pre-Season Testing', 'Pre-Season Test']
        df = df[~df['GP'].isin(excluded_gp)]
        df['Year'] = df['Season']
        return df
