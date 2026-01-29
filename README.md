# ❤️Heart Disease Scoring (FastAPI + Streamlit)

Сервис для предсказания риска сердечного заболевания по медицинским признакам.  
После EDA и бенчмарка моделей выбрал Random Forest по метрикам качества, затем оптимизировал порог под Recall.

## Локальный запуск

```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
python train.py
uvicorn service:app --reload
```
Swagger UI: http://127.0.0.1:8000/docs

Во втором терминале:
```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
streamlit run app.py
```
UI: http://localhost:8501
