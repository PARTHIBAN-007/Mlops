import logging
import yaml
import mlflow
import mlflow.sklearn
from steps.data_ingestion import DataIngestion
from steps.data_preprocessing import Cleaner
from steps.data_train import Trainer
from steps.data_predict import Predictor
from sklearn.metrics import classification_report

logging.basicConfig(level = logging.INFO , format = '%(asctime)s:%(levelname)s:%(message)s')

def main():
    ingestion = DataIngestion()
    train , test = ingestion.load_data()
    logging.inof("Data Ingestion Completed Successfully")

    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data = cleaner.clean_data(test)
    logging.info("Data Cleaning completed Successfully")

    trainer  = Trainer()
    x_train , y_train  = trainer.feature_target_separator(train_data)
    trainer.train_model(x_train,y_train)
    trainer.save_model()
    logging.info("Model training completed Successfully")

    predictor = Predictor()
    x_test , y_test = predictor.feature_target_separator(test_data)
    accuracy , class_report , roc_auc_score = predictor.evaluate_model(x_test,y_test)
    logging.info("Model Evaluation completed successfully")

    print("\n================Model Evaluation results================")
    print(f"Model : {trainer.model_name}")
    print(f"Accuracy score : {accuracy:.4f}, ROC AUC Score : {roc_auc_score}")
    print(f"\n{class_report}")
    print("================================")

def train_with_mlflow():
    with open('config.yml','r') as file:
        config = yaml.safe_load(file)
    
    mlflow.set_experiment("Model Training Experiment")

    with mlflow.start_run() as run:
        ingestion = DataIngestion()
        train , test = ingestion.load_data()
        logging.info("Data Ingestion Completed Successfully")

        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data Cleaning Completed Successfully")


        trainer = Trainer()
        x_train , y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(x_train,y_train)
        trainer.save_model("Model training Completed Successfully")

        predictor = Predictor()
        x_test , y_test = predictor.feature_target_separator(test_data)
        accuracy , class_report , roc_auc_score = predictor.evaluate_model(x_test,y_test)
        report = classification_report(y_test,trainer.pipeline.predict(x_test),output_dict=True)
        logging.info("Model Evaluation completed Successfully")

        mlflow.set_tag('Model Developer','Parthiban K')
        mlflow.set_tag('preprocessing','oneHotEncoder,Standard Scaler,MinMaxScaler')


        model_params = config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("roc",roc_auc_score)
        mlflow.log_metric("precision",report['weighted avg']['precision'])
        mlflow.log_metric('recall', report['weighted avg']['recall'])
        mlflow.sklearn.log_model(trainer.pipeline, "model")


        model_name = "Insurance Prediction"
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri,model_name)

        logging.info("Mlflow Tracking completed Successfully")

        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
        print(f"\n{class_report}")
        print("=====================================================\n")
        
if __name__ == "__main__":
    # main()
    train_with_mlflow()

