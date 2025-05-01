# **Insurance Cross-Sell Prediction with MLOps**

## **Overview**

A machine learning model to predict the likelihood of a customer purchasing additional insurance. A Decision Tree is used for its interpretability,
and MLOps principles are integrated to automate the deployment, monitoring, and scaling of the model.

<img src="Assets\DataDrift.png" >


## **Tech Stack & Features**
- **SMOTE** – Handles class imbalance
- **Decision Tree** – Ensures interpretability
- **MLflow** – Tracks experiments & models
- **FastAPI** – Serves the model via API
- **Streamlit** – Provides a user-friendly UI
- **Docker** – Containerizes the application
- **GitHub Actions** – Automates CI/CD for testing, training & deployment
- **Evidently AI** – Monitors data drift

## **Project Setup**
You can set up and run this project **with** or **without Docker**.

### **1. Setup Without Docker**
#### **Prerequisites**
- Python 3.8+
- pip
- Virtual environment (optional but recommended)

#### **Installation Steps**
1 Clone the repository

```
git https://github.com/PARTHIBAN-007/Mlops
cd Mlops
```
2 Create and activate virtual environment
```sh
python -m venv venv
sourcevenv\Scripts\activate 
```
3 Install dependencies
```
pip install -r requirements.txt
```
4. Data Ingestion, Cleaning and Training the Model
```
python main.py
```
Run MLflow tracking server (optional)
```
mlflow ui
```
4. Start FastAPI server
```
uvicorn app:app.py
```
5. Run Streamlit UI
```
streamlit run streamlit.py
```
Data Drift Monitoring (optional)
```
Run the cells in the monitor.ipynb to access the test_drift.html
```


### **2. Setup With Docker**
#### **Prerequisites**
- Docker installed on your system

#### **Build and Run**
```sh
# Clone the repository
git clone https://github.com/PARTHIBAN-007/Mlops
cd Mlops

# Build the Docker image
docker build -t insurance-prediction .

# Run the container
docker run -p 8000:8000 -p 8501:8501 insurance-prediction
```
<h1>Mlflow Metrics</h1>
<img src="Assets\MlflowMetrics.png" height="500">
<h1>Docker Images</h1>
<img src="Assets\DockerImages.png" height="300" >
<h1>FastAPI</h1>
<img src="Assets\FastAPI.png" height="400">
