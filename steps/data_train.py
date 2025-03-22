import os
import joblib
import yaml
from sklearn.preprocessing import StandardScaler , OneHotEncoder ,MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import pipeline
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


class Trainer:
    def __init__(self):
        self.config = self.load_confi()
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open('config.yaml','r') as config_file:
            return yaml.safe_load(config_file)
        
    def create_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ("minmax",MinMaxScaler(),['AnnualPremium']),
                ('standardize',StandardScaler(),['Age',['RegionID']]),
                ('onehot',OneHotEncoder(handle_unknown="ignore"),['Gender','PastAccident'])

            ]
        )

        smote = SMOTE(sampling_strategy=1.0)

        model_map = {
            "RandomforestClassifier":RandomForestClassifier,
            'DecisonTreeClassifier':DecisionTreeClassifier,
            'GradientBoostingClassifer':GradientBoostingClassifier
        }

        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)

        pipeline = pipeline([
            ('preprocessor',preprocessor)
            ('smote',smote)
            ('model',model)
        ])

        return pipeline
    

    def feature_target_separator(self,data):
        x = data.iloc[:,:-1]
        y = data.iloc[:,-1]

        return x,y
    
    def train_model(self,x_train, y_train):
        self.pipeline.fit(x_train, y_train)

    def save_model(self):
        model_file_path = os.path.join(self.model_path,'model.pkl')
        joblib.dump(self.pipeline,model_file_path)

    