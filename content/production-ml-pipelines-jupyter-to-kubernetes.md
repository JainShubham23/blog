Title: Building Production-Ready ML Pipelines: From Jupyter to Kubernetes
Date: 2025-11-11 14:30
Modified: 2025-11-11 14:30
Category: Programming
Tags: machine-learning, kubernetes, python, devops, mlops, data-science, docker, jupyter, pipeline, production
Slug: production-ml-pipelines-jupyter-to-kubernetes
Authors: Shubham Jain
Summary: A comprehensive guide to transforming your experimental Jupyter notebooks into robust, scalable machine learning pipelines deployed on Kubernetes in production environments.

# Building Production-Ready ML Pipelines: From Jupyter to Kubernetes

Moving machine learning models from experimental Jupyter notebooks to production-ready systems is one of the most challenging aspects of modern data science. While Jupyter notebooks excel at exploration and prototyping, production ML systems require reliability, scalability, monitoring, and maintainability that notebooks simply cannot provide on their own.

In this comprehensive guide, we'll walk through the entire journey of transforming an experimental ML workflow into a robust, production-ready pipeline deployed on Kubernetes. We'll cover everything from code refactoring and containerization to orchestration and monitoring.

## Table of Contents

1. [The Challenge: From Notebook to Production](#the-challenge)
2. [Setting Up the Foundation](#foundation)
3. [Refactoring Jupyter Code](#refactoring)
4. [Building Containerized Services](#containerization)
5. [Creating the ML Pipeline](#pipeline)
6. [Kubernetes Deployment](#kubernetes)
7. [Monitoring and Observability](#monitoring)
8. [Best Practices and Common Pitfalls](#best-practices)
9. [Conclusion](#conclusion)

## The Challenge: From Notebook to Production {#the-challenge}

### Why Notebooks Fall Short in Production

Jupyter notebooks are excellent for:
- **Rapid experimentation** and data exploration
- **Interactive analysis** with immediate visual feedback
- **Collaborative research** and proof-of-concept development
- **Documentation** of analytical thinking process

However, they're problematic for production because:

```python
# Typical notebook code - hard to maintain in production
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Data scattered across multiple cells
df = pd.read_csv('data.csv')
# ... 50 cells later ...
X = df[['feature1', 'feature2', 'feature3']]
# ... another 20 cells ...
model = RandomForestClassifier()
model.fit(X, y)
```

**Production Requirements:**
- **Reproducibility**: Consistent results across environments
- **Scalability**: Handle varying workloads efficiently
- **Reliability**: Robust error handling and recovery
- **Maintainability**: Clean, testable, and modular code
- **Monitoring**: Observability into system health and performance
- **Security**: Proper authentication, authorization, and data protection

## Setting Up the Foundation {#foundation}

### Project Structure

Let's start by establishing a proper project structure that separates concerns and promotes maintainability:

```
ml-pipeline/
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── deployment.yaml
│   └── service.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── training.py
│   │   ├── prediction.py
│   │   └── evaluation.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── schemas.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── monitoring.py
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   └── test_api/
├── notebooks/
│   └── exploration/
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
└── monitoring/
    ├── prometheus/
    └── grafana/
```

### Configuration Management

Create a robust configuration system:

```python
# src/config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    username: str = os.getenv('DB_USER', 'postgres')
    password: str = os.getenv('DB_PASSWORD', '')
    database: str = os.getenv('DB_NAME', 'ml_pipeline')

@dataclass
class ModelConfig:
    name: str = os.getenv('MODEL_NAME', 'random_forest')
    version: str = os.getenv('MODEL_VERSION', '1.0.0')
    max_features: int = int(os.getenv('MODEL_MAX_FEATURES', '10'))
    n_estimators: int = int(os.getenv('MODEL_N_ESTIMATORS', '100'))
    random_state: int = int(os.getenv('MODEL_RANDOM_STATE', '42'))

@dataclass
class APIConfig:
    host: str = os.getenv('API_HOST', '0.0.0.0')
    port: int = int(os.getenv('API_PORT', '8000'))
    workers: int = int(os.getenv('API_WORKERS', '4'))
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')

@dataclass
class Config:
    database: DatabaseConfig
    model: ModelConfig
    api: APIConfig
    
    @classmethod
    def load(cls) -> 'Config':
        return cls(
            database=DatabaseConfig(),
            model=ModelConfig(),
            api=APIConfig()
        )
```

## Refactoring Jupyter Code {#refactoring}

### From Notebook Cells to Modular Functions

Let's transform typical notebook code into production-ready modules:

**Original Notebook Code:**
```python
# Notebook cell 1
import pandas as pd
df = pd.read_csv('customer_data.csv')

# Notebook cell 15
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 100], 
                        labels=['young', 'middle', 'senior', 'elderly'])

# Notebook cell 23
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# Notebook cell 35
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

**Refactored Production Code:**

```python
# src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: list = []
        
    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age group categories."""
        df = df.copy()
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 25, 45, 65, 100], 
            labels=['young', 'middle', 'senior', 'elderly']
        )
        logger.info(f"Created age groups for {len(df)} records")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_columns: list) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        for column in categorical_columns:
            if column not in self.encoders:
                self.encoders[column] = LabelEncoder()
                df[f'{column}_encoded'] = self.encoders[column].fit_transform(df[column])
                logger.info(f"Fitted encoder for {column}")
            else:
                df[f'{column}_encoded'] = self.encoders[column].transform(df[column])
                logger.info(f"Applied existing encoder for {column}")
                
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                                numerical_columns: list, 
                                fit: bool = False) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        if fit:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
            logger.info("Fitted scaler on numerical features")
        else:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
            logger.info("Applied existing scaler to numerical features")
            
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete feature preparation pipeline."""
        logger.info("Starting feature preparation")
        
        # Create derived features
        df = self.create_age_groups(df)
        
        # Encode categorical variables
        categorical_columns = ['gender', 'occupation', 'age_group']
        df = self.encode_categorical_features(df, categorical_columns)
        
        # Scale numerical features
        numerical_columns = ['age', 'income', 'spending_score']
        df = self.scale_numerical_features(df, numerical_columns, fit=True)
        
        # Select final features
        self.feature_columns = (
            [f'{col}_encoded' for col in categorical_columns] + 
            numerical_columns
        )
        
        X = df[self.feature_columns]
        y = df[target_column]
        
        logger.info(f"Prepared {len(X)} samples with {len(self.feature_columns)} features")
        return X, y
```

### Model Training Module

```python
# src/models/training.py
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.best_params = None
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """Train the model with hyperparameter optimization."""
        
        logger.info("Starting model training")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config)
            
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Initialize base model
            base_model = RandomForestClassifier(
                random_state=self.config.get('random_state', 42),
                n_jobs=-1
            )
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, param_grid, 
                cv=5, scoring='f1_weighted',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            # Log best parameters and score
            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_score = self.model.score(X_val, y_val)
                mlflow.log_metric("validation_accuracy", val_score)
                
                y_pred = self.model.predict(X_val)
                report = classification_report(y_val, y_pred, output_dict=True)
                
                # Log detailed metrics
                mlflow.log_metric("validation_precision", report['weighted avg']['precision'])
                mlflow.log_metric("validation_recall", report['weighted avg']['recall'])
                mlflow.log_metric("validation_f1", report['weighted avg']['f1-score'])
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            logger.info(f"Training completed. Best CV score: {grid_search.best_score_:.4f}")
            
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
```

## Building Containerized Services {#containerization}

### Dockerfile for ML Service

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash mluser
USER mluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "src.api.app"]
```

### FastAPI Application

```python
# src/api/app.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import joblib
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time

from ..config import Config
from ..data.preprocessing import DataPreprocessor
from ..utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('ml_predictions_total', 'Total predictions made')
prediction_latency = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
error_counter = Counter('ml_prediction_errors_total', 'Total prediction errors')

app = FastAPI(
    title="ML Prediction Service",
    description="Production ML service for customer segmentation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
preprocessor = None
config = None

class PredictionRequest(BaseModel):
    age: int
    gender: str
    occupation: str
    income: float
    spending_score: int

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    request_id: str

@app.on_event("startup")
async def startup_event():
    """Initialize model and preprocessor on startup."""
    global model, preprocessor, config
    
    config = Config.load()
    
    try:
        # Load model
        model = joblib.load('/app/models/model.pkl')
        preprocessor = joblib.load('/app/models/preprocessor.pkl')
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction for a single sample."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Preprocess data
        X_processed, _ = preprocessor.prepare_features(
            input_data, target_column=None
        )
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = float(model.predict_proba(X_processed)[0].max())
        
        # Update metrics
        prediction_counter.inc()
        prediction_latency.observe(time.time() - start_time)
        
        logger.info(f"Prediction made: {prediction} (prob: {probability:.3f})")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            request_id=request_id
        )
        
    except Exception as e:
        error_counter.inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Make predictions for multiple samples."""
    start_time = time.time()
    
    try:
        # Convert requests to DataFrame
        input_data = pd.DataFrame([req.dict() for req in requests])
        
        # Preprocess data
        X_processed, _ = preprocessor.prepare_features(
            input_data, target_column=None
        )
        
        # Make predictions
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed).max(axis=1)
        
        results = [
            {
                "prediction": pred,
                "probability": float(prob),
                "request_id": f"batch_req_{i}_{int(time.time() * 1000)}"
            }
            for i, (pred, prob) in enumerate(zip(predictions, probabilities))
        ]
        
        # Update metrics
        prediction_counter.inc(len(requests))
        prediction_latency.observe(time.time() - start_time)
        
        logger.info(f"Batch prediction completed: {len(requests)} samples")
        
        return {"predictions": results}
        
    except Exception as e:
        error_counter.inc()
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        log_level=config.api.log_level.lower()
    )
```

## Creating the ML Pipeline {#pipeline}

### Pipeline Orchestration with Airflow

```python
# dags/ml_pipeline_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.sensors.s3_key_sensor import S3KeySensor

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 11),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='ML model training and deployment pipeline',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1
)

# Data validation task
validate_data = PythonOperator(
    task_id='validate_data',
    python_callable=validate_input_data,
    dag=dag
)

# Data preprocessing task
preprocess_data = DockerOperator(
    task_id='preprocess_data',
    image='ml-pipeline:latest',
    command='python -m src.data.preprocessing',
    environment={
        'DATA_PATH': '/data/raw/customer_data.csv',
        'OUTPUT_PATH': '/data/processed/features.csv'
    },
    dag=dag
)

# Model training task
train_model = DockerOperator(
    task_id='train_model',
    image='ml-pipeline:latest',
    command='python -m src.models.training',
    environment={
        'FEATURES_PATH': '/data/processed/features.csv',
        'MODEL_OUTPUT_PATH': '/models/trained_model.pkl'
    },
    dag=dag
)

# Model evaluation task
evaluate_model = DockerOperator(
    task_id='evaluate_model',
    image='ml-pipeline:latest',
    command='python -m src.models.evaluation',
    dag=dag
)

# Model deployment task
deploy_model = BashOperator(
    task_id='deploy_model',
    bash_command='''
    kubectl set image deployment/ml-service \
    ml-service=ml-pipeline:{{ ds_nodash }} \
    -n ml-production
    ''',
    dag=dag
)

# Set up dependencies
validate_data >> preprocess_data >> train_model >> evaluate_model >> deploy_model
```

## Kubernetes Deployment {#kubernetes}

### Kubernetes Manifests

**Namespace Configuration:**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-production
  labels:
    name: ml-production
```

**ConfigMap for Application Configuration:**
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-service-config
  namespace: ml-production
data:
  MODEL_NAME: "customer_segmentation"
  MODEL_VERSION: "1.0.0"
  LOG_LEVEL: "INFO"
  API_WORKERS: "4"
  DB_HOST: "postgresql-service"
  DB_PORT: "5432"
  DB_NAME: "ml_pipeline"
```

**Deployment Configuration:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: ml-production
  labels:
    app: ml-service
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: ml-service
        image: ml-pipeline:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: API_HOST
          value: "0.0.0.0"
        - name: API_PORT
          value: "8000"
        envFrom:
        - configMapRef:
            name: ml-service-config
        - secretRef:
            name: ml-service-secrets
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      imagePullSecrets:
      - name: registry-secret
```

**Service Configuration:**
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-service
  namespace: ml-production
  labels:
    app: ml-service
spec:
  selector:
    app: ml-service
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: ml-service-nodeport
  namespace: ml-production
  labels:
    app: ml-service
spec:
  selector:
    app: ml-service
  ports:
  - name: http
    port: 80
    targetPort: 8000
    nodePort: 30080
    protocol: TCP
  type: NodePort
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
  namespace: ml-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

## Monitoring and Observability {#monitoring}

### Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ml_service_rules.yml"

scrape_configs:
  - job_name: 'ml-service'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - ml-production
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      target_label: __address__
    - action: labelmap
      regex: __meta_kubernetes_pod_label_(.+)

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Custom Monitoring Metrics

```python
# src/utils/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, Info
import psutil
import time
import logging

logger = logging.getLogger(__name__)

# Application metrics
prediction_requests = Counter(
    'ml_prediction_requests_total',
    'Total prediction requests',
    ['method', 'endpoint', 'status']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction request latency',
    ['method', 'endpoint']
)

model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy score'
)

data_drift_score = Gauge(
    'ml_data_drift_score',
    'Data drift detection score'
)

# System metrics
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')

# Model information
model_info = Info(
    'ml_model_info',
    'Information about the current model'
)

class MetricsCollector:
    """Collect and update system and application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            cpu_usage.set(psutil.cpu_percent())
            memory_usage.set(psutil.virtual_memory().percent)
            disk_usage.set(psutil.disk_usage('/').percent)
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def update_model_info(self, name: str, version: str, accuracy: float):
        """Update model information metrics."""
        model_info.info({
            'name': name,
            'version': version,
            'training_date': time.strftime('%Y-%m-%d'),
            'framework': 'scikit-learn'
        })
        model_accuracy.set(accuracy)
        logger.info(f"Updated model metrics: {name} v{version} (acc: {accuracy:.3f})")

    def record_prediction(self, method: str, endpoint: str, 
                         duration: float, status: str):
        """Record prediction request metrics."""
        prediction_requests.labels(
            method=method, 
            endpoint=endpoint, 
            status=status
        ).inc()
        
        prediction_latency.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "ML Service Dashboard",
    "tags": ["ml", "production"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Prediction Requests Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(ml_prediction_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "palette-classic"
            },
            "custom": {
              "displayMode": "list",
              "orientation": "horizontal"
            }
          }
        }
      },
      {
        "title": "Prediction Latency",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, ml_prediction_latency_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, ml_prediction_latency_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "gauge",
        "targets": [
          {
            "expr": "ml_model_accuracy",
            "legendFormat": "Accuracy"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 0.7},
                {"color": "green", "value": 0.85}
              ]
            }
          }
        }
      },
      {
        "title": "System Resources",
        "type": "timeseries",
        "targets": [
          {
            "expr": "system_cpu_usage_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "system_memory_usage_percent",
            "legendFormat": "Memory %"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

## Best Practices and Common Pitfalls {#best-practices}

### Code Quality and Testing

```python
# tests/test_models/test_training.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.models.training import ModelTrainer

class TestModelTrainer:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'random_state': 42,
            'n_estimators': 10,  # Smaller for faster tests
            'max_depth': 5
        }
        self.trainer = ModelTrainer(self.config)
        
        # Create sample data
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randint(0, 2, 100)
        })
        self.y_train = pd.Series(np.random.randint(0, 3, 100))
        
    def test_train_model_success(self):
        """Test successful model training."""
        with patch('mlflow.start_run'):
            self.trainer.train_model(self.X_train, self.y_train)
            
        assert self.trainer.model is not None
        assert self.trainer.best_params is not None
        
    def test_train_model_with_validation(self):
        """Test training with validation set."""
        X_val = self.X_train.sample(20)
        y_val = self.y_train.sample(20)
        
        with patch('mlflow.start_run'):
            self.trainer.train_model(
                self.X_train, self.y_train, 
                X_val, y_val
            )
            
        assert self.trainer.model is not None
        
    def test_save_load_model(self, tmp_path):
        """Test model saving and loading."""
        model_path = tmp_path / "test_model.pkl"
        
        # Train and save model
        with patch('mlflow.start_run'):
            self.trainer.train_model(self.X_train, self.y_train)
        self.trainer.save_model(str(model_path))
        
        # Load model in new trainer
        new_trainer = ModelTrainer(self.config)
        new_trainer.load_model(str(model_path))
        
        assert new_trainer.model is not None
        
        # Test predictions are identical
        pred1 = self.trainer.model.predict(self.X_train[:10])
        pred2 = new_trainer.model.predict(self.X_train[:10])
        np.testing.assert_array_equal(pred1, pred2)
```

### Data Validation

```python
# src/data/validation.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]

class DataValidator:
    """Validate input data for ML pipeline."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Comprehensive dataframe validation."""
        errors = []
        warnings = []
        statistics = {}
        
        # Check required columns
        missing_cols = set(self.schema['required_columns']) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            
        # Check data types
        for col, expected_type in self.schema['column_types'].items():
            if col in df.columns:
                if not df[col].dtype == expected_type:
                    try:
                        df[col] = df[col].astype(expected_type)
                        warnings.append(f"Converted {col} to {expected_type}")
                    except:
                        errors.append(f"Cannot convert {col} to {expected_type}")
                        
        # Check for missing values
        missing_stats = df.isnull().sum()
        statistics['missing_values'] = missing_stats.to_dict()
        
        for col, missing_count in missing_stats.items():
            missing_pct = missing_count / len(df) * 100
            if missing_pct > self.schema['max_missing_percentage'].get(col, 10):
                errors.append(
                    f"Column {col} has {missing_pct:.1f}% missing values "
                    f"(max allowed: {self.schema['max_missing_percentage'].get(col, 10)}%)"
                )
                
        # Check value ranges
        for col, (min_val, max_val) in self.schema.get('value_ranges', {}).items():
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                if out_of_range.any():
                    count = out_of_range.sum()
                    warnings.append(
                        f"Column {col} has {count} values outside range [{min_val}, {max_val}]"
                    )
                    
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")
            
        # Collect basic statistics
        statistics.update({
            'row_count': len(df),
            'column_count': len(df.columns),
            'duplicate_rows': duplicate_count,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        })
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )

# Example schema configuration
CUSTOMER_DATA_SCHEMA = {
    'required_columns': ['age', 'gender', 'income', 'occupation'],
    'column_types': {
        'age': 'int64',
        'gender': 'object',
        'income': 'float64',
        'occupation': 'object',
        'spending_score': 'int64'
    },
    'value_ranges': {
        'age': (18, 100),
        'income': (0, 1000000),
        'spending_score': (1, 100)
    },
    'max_missing_percentage': {
        'age': 0,
        'gender': 5,
        'income': 10,
        'occupation': 15
    }
}
```

### Security Best Practices

```python
# src/api/security.py
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from typing import Optional
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()

class SecurityManager:
    """Handle authentication and authorization."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
        """Verify JWT token and return user information."""
        try:
            payload = jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
            
    def require_permission(self, required_permission: str):
        """Decorator to require specific permission."""
        def permission_checker(user_info: dict = Depends(self.verify_token)) -> dict:
            user_permissions = user_info.get('permissions', [])
            if required_permission not in user_permissions:
                raise HTTPException(
                    status_code=403,
                    detail=f"Required permission: {required_permission}"
                )
            return user_info
        return permission_checker

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# Apply to FastAPI app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Usage in endpoints
@app.post("/predict")
@limiter.limit("100/minute")
async def predict_with_rate_limit(
    request: Request,
    prediction_request: PredictionRequest,
    user_info: dict = Depends(security_manager.require_permission("ml:predict"))
):
    # Prediction logic here
    pass
```

### Common Pitfalls and Solutions

**1. Model Version Management**
```python
# src/models/versioning.py
import hashlib
import json
from pathlib import Path
from typing import Dict, Any

class ModelVersionManager:
    """Manage model versions and metadata."""
    
    def __init__(self, model_registry_path: str):
        self.registry_path = Path(model_registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
    def calculate_model_hash(self, model_path: str, 
                           hyperparameters: Dict[str, Any]) -> str:
        """Calculate unique hash for model version."""
        with open(model_path, 'rb') as f:
            model_content = f.read()
            
        # Combine model binary and hyperparameters
        combined = model_content + json.dumps(hyperparameters, sort_keys=True).encode()
        return hashlib.sha256(combined).hexdigest()[:16]
        
    def register_model(self, model_path: str, metadata: Dict[str, Any]) -> str:
        """Register a new model version."""
        model_hash = self.calculate_model_hash(model_path, metadata['hyperparameters'])
        
        # Create version directory
        version_dir = self.registry_path / f"v{model_hash}"
        version_dir.mkdir(exist_ok=True)
        
        # Copy model file
        import shutil
        shutil.copy2(model_path, version_dir / "model.pkl")
        
        # Save metadata
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump({
                **metadata,
                'model_hash': model_hash,
                'registration_time': datetime.now().isoformat()
            }, f, indent=2)
            
        return model_hash
```

**2. Data Drift Detection**
```python
# src/monitoring/drift_detection.py
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DataDriftDetector:
    """Detect data drift in production."""
    
    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        
    def kolmogorov_smirnov_test(self, new_data: np.ndarray) -> Tuple[float, bool]:
        """Perform KS test for drift detection."""
        statistic, p_value = stats.ks_2samp(self.reference_data, new_data)
        is_drift = p_value < self.threshold
        
        logger.info(f"KS test: statistic={statistic:.4f}, p_value={p_value:.4f}, drift={is_drift}")
        return p_value, is_drift
        
    def population_stability_index(self, new_data: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(self.reference_data, bins=bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(self.reference_data, bins=bin_edges)
        new_counts, _ = np.histogram(new_data, bins=bin_edges)
        
        # Convert to percentages and avoid division by zero
        ref_pcts = ref_counts / len(self.reference_data) + 1e-10
        new_pcts = new_counts / len(new_data) + 1e-10
        
        # Calculate PSI
        psi = np.sum((new_pcts - ref_pcts) * np.log(new_pcts / ref_pcts))
        
        logger.info(f"PSI: {psi:.4f}")
        return psi
```

## Conclusion {#conclusion}

Building production-ready ML pipelines requires a fundamental shift from the experimental mindset of Jupyter notebooks to the engineering rigor of production systems. The journey from notebook to Kubernetes involves several critical transformations:

### Key Takeaways

1. **Code Organization**: Transform notebook cells into modular, testable functions with clear separation of concerns.

2. **Configuration Management**: Use environment-specific configurations to handle different deployment scenarios.

3. **Containerization**: Package your ML services as Docker containers for consistency across environments.

4. **Orchestration**: Leverage Kubernetes for scalable, resilient deployments with proper resource management.

5. **Monitoring**: Implement comprehensive observability to track both system health and model performance.

6. **Data Quality**: Establish robust data validation and drift detection to ensure model reliability.

7. **Security**: Apply proper authentication, authorization, and rate limiting to protect your services.

### Production Readiness Checklist

Before deploying your ML pipeline to production, ensure you have:

- ✅ **Modular code structure** with proper error handling
- ✅ **Comprehensive tests** for all critical components  
- ✅ **CI/CD pipeline** for automated testing and deployment
- ✅ **Monitoring and alerting** for system and model metrics
- ✅ **Data validation** and drift detection mechanisms
- ✅ **Security measures** including authentication and rate limiting
- ✅ **Documentation** for APIs, deployment, and troubleshooting
- ✅ **Rollback strategy** for failed deployments
- ✅ **Resource limits** and autoscaling configuration
- ✅ **Backup and recovery** procedures

### Next Steps

The ML operations (MLOps) journey doesn't end with deployment. Consider implementing:

- **A/B testing frameworks** for model comparison
- **Feature stores** for consistent feature engineering
- **Model explanation** and interpretability tools
- **Automated retraining** pipelines based on performance metrics
- **Multi-model serving** for ensemble approaches
- **Edge deployment** for latency-critical applications

Building production ML systems is an iterative process that requires collaboration between data scientists, ML engineers, and DevOps teams. Start small, iterate frequently, and continuously improve your pipeline based on real-world performance and user feedback.

The transformation from Jupyter to Kubernetes represents more than just a technical migration—it's a maturation from research to engineering, from prototype to product, from experiment to experience.

---

*Have you successfully deployed ML models to production? Share your experiences and challenges in the comments below. For more technical deep-dives, follow my blog for regular updates on MLOps, data engineering, and software development best practices.*