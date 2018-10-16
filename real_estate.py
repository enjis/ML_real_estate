import numpy as np
import pandas as pd
#pd.set_option('display.max_colwidth',100,'display.max_columns',999)
#pd.set_option('mode.chained_assignment',None)
import pickle
import sklearn
import sys


class RealEstateModel:
    
    def __init__(self, model_location):
        with open(model_location, 'rb') as f:
            self.model = pickle.load(f)
            
    def predict(self, X_new, clean=True, augment=True):
        if clean:
            X_new = self.clean_data(X_new)
            
        if augment:
            X_new = self.engineer_features(X_new)
            
        return X_new, self.model.predict(X_new)
    
    def clean_data(self, df):
        df.drop_duplicates()
        df['basement'] = df.basement.fillna(0)
        df.roof.replace('composition','Composition',inplace=True)
        df.roof.replace('asphalt','Asphalt',inplace=True)
        df.roof.replace(['shake-shingle','asphalt,shake-shingle'],'Shake Shingle',inplace=True)
        df.exterior_walls.replace('Rock, Stone','Masonry',inplace=True)
        df.exterior_walls.replace(['Concrete','Block'],'Concrete Block',inplace=True)
        df = df[df.lot_size<=500000]
        for column in df.select_dtypes(include=['object']):
            df[column] = df[column].fillna('Missing')
        return df
    
    def engineer_features(self, df):
        df['two_two'] = ((df.beds==2)& (df.baths==2)).astype(int)
        df['during_recession'] = df.tx_year.between(2010,2013).astype(int)
        df['property_age'] = df.tx_year - df.year_built
        df = df[df.property_age >=0]
        df['school_score'] = df.num_schools * df.median_school
        df.exterior_walls.replace(['Wood Siding','Wood Shingle','Wood'],'Wood',inplace=True)
        other_exterior_walls=['Stucco','Other','Asbestos shingle','Concrete Block','Masonry']
        df.exterior_walls.replace(other_exterior_walls,'Other',inplace=True)
        df.roof.replace(['Composition','Wood Shake/ Shingles'],'Composition Shingle',inplace=True)
        other_roofs=['Other','Gravel/Rock','Roll Composition','Slate','Built-up','Asbestos','Metal']
        df.roof.replace(other_roofs,'Other',inplace=True)
        df = pd.get_dummies(df,columns=['exterior_walls','roof','property_type'])
        df = df.drop(['tx_year','year_built'],axis=1)
        return df
    
def main(data_location, output_location, model_location, clean=True, augment=True):
    df = pd.read_csv(data_location)
    estate_model = RealEstateModel(model_location)
    df, pred = estate_model.predict(df)
    df['predicted tx_price'] = pred
    df.to_csv(output_location, index=None)

if __name__ == '__main__':
    main(*sys.argv[1:])