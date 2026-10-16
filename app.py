import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
SEX = {1: "Женский", 2: "Мужской"}
CHOLESTEROL = {None: "Не знаю", 1: "<200 мг/дл", 2: "200–<240 мг/дл", 3: "≥240 мг/дл"}
HBA1C = {None: "Не знаю", 1: "<5.7%", 2: "5.7–<6.5%", 3: "≥6.5%"}
YES_NO = {None: "Не знаю", 0: "Нет", 1: "Да"}
NAMES = {"height": "рост", "weight": "вес", "ap_hi": "верхнее давление",
         "ap_lo": "нижнее давление", "cholesterol": "холестерин", "gluc": "HbA1c",
         "smoke": "курение", "alco": "алкоголь", "active": "физическая активность"}

st.set_page_config(page_title="Heart Disease Scoring", page_icon="🫀", layout="centered")
st.title("🫀 Оценка показателей здоровья")
st.caption("Учебный проект: результат помогает обсудить показатели, но не заменяет консультацию врача.")


def show_result(result):
    st.subheader("Результат")
    if result["above_threshold"]:
        st.warning("Оценка выше порога модели")
    else:
        st.info("Оценка ниже порога модели")
    st.write(result["recommendation"])
    st.caption(result["notice"])

    if result["imputed_fields"]:
        fields = ", ".join(NAMES[name] for name in result["imputed_fields"])
        st.info(f"Не указаны: {fields}. При расчёте применены правила заполнения из обучения. "
                "Если значения известны, внесите их и повторите оценку.")

    with st.expander("Оценка и качество модели"):
        c1, c2 = st.columns(2)
        c1.metric("Оценка модели · 0–1", f"{result['score']:.3f}")
        c2.metric("Порог решения", f"{result['threshold']:.3f}")
        st.write(f"Модель: **{result['model']}**. Оценку нельзя читать как процент личного риска.")
        scores = result["test_metrics"]
        st.write(f"На тестовой выборке обнаружено **{scores['Recall']:.1%}** положительных случаев. "
                 f"Среди положительных ответов модели верными оказались **{scores['Precision']:.1%}**.")
        st.caption(f"AP = {scores['AP']:.3f}; ROC-AUC = {scores['ROC_AUC']:.3f}. "
                   "Это проверка на части одного датасета; клиническая применимость не подтверждена.")


with st.form("heart_form"):
    st.subheader("Данные обследования")
    st.caption("Возраст и пол обязательны. Неизвестные показатели оставьте пустыми или выберите «Не знаю».")
    st.caption("Давление укажите парой. Рост и вес также заполните вместе или оставьте оба поля пустыми.")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Возраст, лет", min_value=20, max_value=79, value=None, step=1)
        gender = st.selectbox("Пол", list(SEX), index=None, format_func=SEX.get, placeholder="Выберите пол")
        height = st.number_input("Рост, см", 120.0, 220.0, value=None, step=1.0)
        weight = st.number_input("Вес, кг", 30.0, 200.0, value=None, step=0.5)
    with c2:
        ap_hi = st.number_input("Верхнее давление, мм рт. ст.", 70, 250, value=None)
        ap_lo = st.number_input("Нижнее давление, мм рт. ст.", 40, 150, value=None)
        cholesterol = st.selectbox("Общий холестерин", list(CHOLESTEROL), format_func=CHOLESTEROL.get)
        gluc = st.selectbox("HbA1c (гликированный гемоглобин)", list(HBA1C), format_func=HBA1C.get)
    c1, c2, c3 = st.columns(3)
    smoke = c1.selectbox("Курение", list(YES_NO), format_func=YES_NO.get)
    alco = c2.selectbox("Употребление алкоголя", list(YES_NO), format_func=YES_NO.get)
    active = c3.selectbox("Физическая активность", list(YES_NO), format_func=YES_NO.get)
    submitted = st.form_submit_button("Получить оценку", type="primary")

if submitted:
    if age is None or gender is None:
        st.error("Укажите возраст и пол.")
    elif (ap_hi is None) != (ap_lo is None):
        st.error("Укажите оба значения давления или ни одного.")
    elif (height is None) != (weight is None):
        st.error("Укажите рост и вес вместе или оставьте оба поля пустыми.")
    elif ap_hi is not None and ap_lo is not None and ap_hi <= ap_lo:
        st.error("Верхнее давление должно быть выше нижнего.")
    else:
        payload = {"age": age, "gender": gender, "height": height, "weight": weight,
                   "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol, "gluc": gluc,
                   "smoke": smoke, "alco": alco, "active": active}
        try:
            with st.spinner("Рассчитываем оценку…"):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if response.status_code == 422:
                st.error("API отклонил данные. Проверьте диапазоны значений и категории.")
            else:
                response.raise_for_status()
                show_result(response.json())
        except requests.RequestException:
            st.error("API недоступен или вернул ошибку. Проверьте запуск FastAPI и адрес API_URL.")
        except (ValueError, KeyError, TypeError):
            st.error("Формат ответа API не совпадает с интерфейсом. Используйте файлы одной версии проекта.")
