📧 AI Spam Email Classifier
A Modern Streamlit Web App for Classifying Emails as Spam or Ham
🚀 Overview

This project is a clean, responsive, and modern Streamlit-based web application that uses a machine learning model to classify email text into:

🚨 SPAM — phishing, scam, promotional, or malicious email
✅ HAM — legitimate and safe email

The app includes smooth animations, gradient UI, example text buttons, and probability-based predictions.

🧠 Features
✔️ Modern Glassmorphism UI
✔️ Responsive Layout (Desktop + Mobile)
✔️ Upload-free, paste-based detection
✔️ Spam/Ham Probability Metrics
✔️ Interactive Example Email Loader
✔️ Scikit-learn ML Pipeline
✔️ Custom CSS Styling
✔️ Fast & Lightweight

📂 Project Structure
project/
│── app.py
│── spam_classifier_pipeline.pkl
│── requirements.txt
│── README.md
└── spam_classification.ipynb   # Training notebook

📦 Installation & Setup
1. Clone the Repository
git clone https://github.com/your-username/spam_vs_ham.git
cd repository-name

2. Create Virtual Environment
python -m venv venv

3. Activate Virtual Environment

Windows:

venv\Scripts\activate


Mac/Linux:
source venv/bin/activate

4. Install Requirements
pip install -r requirements.txt

5. Run the Application
streamlit run app.py


The app will launch at
👉 http://localhost:8501
📦 Requirements File

Requirements.txt:
streamlit==1.40.2
numpy==1.26.4
scikit-learn==1.5.2
pandas==2.2.3
pickle5==0.0.11

🧪 Model Training Notebook
The spam classification model was trained using the notebook:
📄 spam_classification.ipynb
(Located in the project root)
This notebook includes:
Data preprocessing
Text vectorization
Model training
Pipeline creation
Pickle export

🔮 How the Model Works
User pastes email text
Text is preprocessed & vectorized
Scikit-learn model predicts class (0=HAM, 1=SPAM)
Probability scores are displayed visually
UI highlights whether the message is safe or suspicious

🌍 Deployment Options
Streamlit Cloud (Recommended)
Just upload your repository and set:
streamlit run app.py
Render Deployment

Use a Procfile:
web: streamlit run app.py --server.port=8000 --server.address=0.0.0.0
Docker Deployment
If needed, I can generate a complete Dockerfile for you.

🤝 Contributing
You are welcome to open issues and submit PRs.
Feel free to enhance UI, model performance, or features.

📜 License
This project is available for educational and personal use.

⭐ Support
If you found this useful, please ⭐ star the repository!
