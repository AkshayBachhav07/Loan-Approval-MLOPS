Loan Approval Prediction System (End-to-End MLOps)An enterprise-grade, end-to-end Machine Learning Operations (MLOps) microservice designed to automate retail loan eligibility assessment. The system evaluates applicant financial health, debt-to-income balance, credit track record, and employment standing to deliver fast, consistent, and explainable lending decisions.Built with Python and Scikit-learn, served asynchronously via FastAPI, containerized with Docker, validated with Pytest, and managed through an automated GitHub Actions CI/CD pipeline.📌 Repository Architecture & Directory StructureLoan-Approval-MLOPS/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions workflow (test, train, Docker build)
├── data/
│   └── dataset.csv            # Tabular loan applicant dataset
├── models/
│   └── model.pkl              # Serialized Scikit-learn Pipeline (Scaler + GradientBoosting)
├── app.py                     # FastAPI REST microservice with Pydantic v2 schemas
├── train.py                   # Data generator, feature pipeline & model training script
├── test_app.py                # Automated Pytest suite for API endpoints & model logic
├── requirements.txt           # Pinned production and test dependencies
├── Dockerfile                 # Multi-stage production container specification
├── .gitignore                 # Environment, cache, and artifact exclusions
└── README.md                  # System documentation & execution guide
🛠️ Technology StackProgramming Language: Python 3.11Machine Learning: Scikit-learn (GradientBoostingClassifier), Pandas, NumPy, JoblibAPI Framework: FastAPI, Uvicorn, Pydantic v2Quality Assurance & Testing: Pytest, HTTPXDevOps & Containerization: Docker, GitHub Actions, Render Cloud📊 Dataset Schema & Decision FeaturesThe underlying machine learning model evaluates key quantitative risk indicators:Feature NameData TypePermissible Range / ValuesDescriptionapplicant_incomeFloat₹5,000 – ₹1,000,000Primary monthly income of applicantcoapplicant_incomeFloat₹0 – ₹1,000,000Secondary co-signer monthly incomeloan_amountFloat₹10,000 – ₹5,000,000Requested total loan principal amountcredit_historyInteger0 or 1Past credit track record (1 = Clear, 0 = Defaulted)employment_statusInteger0 or 1Formal employment status (1 = Employed, 0 = Unemployed)existing_loans_countInteger0 – 15Number of active outstanding debt linesloan_status (Target)Integer0 (Reject) or 1 (Approve)Underwriting qualification decision🚀 Local Setup & Quickstart1. Clone the Repositorygit clone https://github.com/AkshayBachhav07/Loan-Approval-MLOPS.git
cd Loan-Approval-MLOPS
2. Set Up Virtual Environment# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Required Dependenciespython -m pip install --upgrade pip
pip install -r requirements.txt
4. Train & Export the Machine Learning PipelineExecutes data generation, applies stratified train-test splitting, fits a standard scaler and gradient boosting ensemble, and saves the binary pipeline artifact to models/model.pkl:python train.py
5. Execute Automated Test Suitepytest test_app.py -v
6. Start the Local API Serverpython -m uvicorn app:app --reload --reload-dir .
Access the interactive OpenAPI Swagger UI documentation at:
👉 http://127.0.0.1:8000/docs🐳 Docker Deployment1. Build the Docker Imagedocker build -t loan-approval-api:latest .
2. Run the Containerized Servicedocker run -d -p 8000:8000 --name loan-service loan-approval-api:latest
Verify service availability by visiting http://localhost:8000/health.📡 API Contract & Endpoints1. Health Probe EndpointMethod: GETRoute: /healthResponse:{
  "status": "healthy",
  "model_artifact_loaded": true
}
2. Loan Approval Inference EndpointMethod: POSTRoute: /predictSample Payload:{
  "applicant_income": 85000.0,
  "coapplicant_income": 30000.0,
  "loan_amount": 150000.0,
  "credit_history": 1,
  "employment_status": 1,
  "existing_loans_count": 0
}
Sample Output:{
  "loan_approved": 1,
  "decision": "Approved",
  "approval_probability": 0.9412,
  "status": "Success"
}
