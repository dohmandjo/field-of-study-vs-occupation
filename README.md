# 🎓 Field of Study vs Occupation - Career Change Prediction

This is a full-stack data engineering and machine learning project that predicts whether individuals are likely to change careers based on their background. It uses PySpark data exploration and modeling, Django as the main backend (with SQLite), and FastAPI as a prediction microservice. The frontend is built using Vite and Tailwind CSS.


## 📁 Project Structure

/1st-DE-Project-FieldOfStudy-Occupation
├── career_prediction_model/
│ └── data/ ← Saved PySpark ML model and preprocessors
│ └── metadata/
├──Data/
│ └── raw/
│ └── cleaned/
├──node_modules
├── notebooks/
│ └── 01_data_loading_and_exploration.ipynb
│ └── 02_modeling_career_change.ipynb
├── ui/
│ ├── api/ ← FastAPI backend
│ ├── api_client/ ← Django API client for the FastAPI backend
│ ├── career_predictor_ui/ ← Vite + Tailwind CSS frontend
├── README.md
└── requirements.txt


---

## 🧠 Objective

The project aims to provide insights into career change likelihood based on educational background and other features, support career counselors and analysts in identifying patterns in career transitions, and improve career counseling services by enabling better guidance for individuals considering a career shift.

---

## 🚀 Key Features

- Logistic Regression model trained with PySpark
- FastAPI backend to expose model predictions
- Django client handling data interaction and history logging
- Vite + Tailwind CSS for an interactive, modern UI
- Clear model metadata: probability scores & top influencing features

---

## 🔧 Tech Stack

- **PySpark** — machine learning & data transformation
- **FastAPI** — API backend for model serving
- **Django** — frontend-facing API client with SQLite3 DB
- **Vite + Tailwind CSS** — UI/UX
- **Pandas** — used in data prep notebooks

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/1st-DE-Project-FieldOfStudy-Occupation.git
cd 1st-DE-Project-FieldOfStudy-Occupation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Django Backend (with SQLite3)

```bash
cd ui/backend
python manage.py migrate
python manage.py runserver
```
Available at: http://localhost:8000/

### 4. Run FastAPI Backend

```bash
cd ui/api
uvicorn main:app --reload
```
API docs at: http://localhost:8001/docs
You may configure ports using .env or uvicorn --port=8001

### 5. Run the Frontend

```bash
cd ui/career_predictor_ui
npm install
npm run dev
```
Frontend UI: http://localhost:5173

## Dataset
- Source: [Kaggle - Field of Study vs Occupation](https://www.kaggle.com/datasets/jahnavipaliwal/field-of-study-vs-occupationg)


## Database

- SQLite3 (for local development)
- Integrated via Django ORM

## License

This project is licensed under the Apache 2.0 License.