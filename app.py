import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Скоринг риска", layout="centered")
st.title("Скоринг риска сердечного заболевания")

SEX = {1: "мужчина", 0: "женщина"}
FBS = {1: "да (> 120 mg/dl)", 0: "нет (≤ 120 mg/dl)"}
EXANG = {1: "да", 0: "нет"}

CP = {
    0: "типичная стенокардия",
    1: "атипичная стенокардия",
    2: "неангинозная боль",
    3: "бессимптомно",
}

RESTECG = {
    0: "норма",
    1: "аномалия ST-T",
    2: "гипертрофия левого желудочка",
}

SLOPE = {
    0: "восходящий (upsloping)",
    1: "плоский (flat)",
    2: "нисходящий (downsloping)",
}

THAL = {
    0: "норма (если встречается)",
    1: "фиксированный дефект",
    2: "обратимый дефект",
    3: "другое/неизвестно (если встречается)",
}


def pick(label, mapping, default, help_text=None):
    keys = list(mapping.keys())
    idx = keys.index(default) if default in keys else 0
    return st.selectbox(
        label,
        keys,
        index=idx,
        format_func=lambda k: f"{k} - {mapping[k]}",
        help=help_text,
    )


with st.form("heart_form"):
    st.subheader("Основные признаки")

    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("age - возраст (лет)", 1, 120, 52)
        trestbps = st.number_input("trestbps - давление в покое (мм рт. ст.)", 50, 250, 125)
        chol = st.number_input("chol - холестерин (мг/дл)", 50, 700, 212)
        thalach = st.number_input("thalach - максимальная ЧСС", 50, 250, 168)
        oldpeak = st.number_input(
            "oldpeak - депрессия ST (нагрузка vs покой)",
            0.0, 10.0, 1.0, step=0.1
        )

    with c2:
        sex = pick("sex - пол", SEX, 1)
        cp = pick("cp - тип боли в груди", CP, 0)
        fbs = pick("fbs - сахар натощак > 120 mg/dl", FBS, 0)
        restecg = pick("restecg - ЭКГ в покое", RESTECG, 0)
        exang = pick("exang - стенокардия при нагрузке", EXANG, 0)

    st.divider()
    st.subheader("Признаки медицинских тестов")

    c3, c4, c5 = st.columns(3)

    with c3:
        slope = pick("slope - наклон ST-сегмента на пике нагрузки", SLOPE, 2)

    with c4:
        ca = st.number_input("ca - количество крупных сосудов (0–3/4)", 0, 4, 0)

    with c5:
        thal = pick("thal - результат thal-теста", THAL, 2)

    submitted = st.form_submit_button("Спрогнозировать")


if submitted:
    payload = {
        "age": int(age),
        "sex": int(sex),
        "cp": int(cp),
        "trestbps": int(trestbps),
        "chol": int(chol),
        "fbs": int(fbs),
        "restecg": int(restecg),
        "thalach": int(thalach),
        "exang": int(exang),
        "oldpeak": float(oldpeak),
        "slope": int(slope),
        "ca": int(ca),
        "thal": int(thal),
    }

    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if r.status_code != 200:
            st.error(f"Ошибка API {r.status_code}: {r.text}")
        else:
            data = r.json()

            p = float(data["p_target"])
            thr = float(data["threshold"])
            pred = int(data["predicted_target"])

            st.subheader("Результат")
            st.metric("Вероятность (target=1)", f"{p:.4f}")

            st.write(f"Порог решения: **{thr:.4f}**")

            if pred == 1:
                st.warning("Стоит понаблюдаться у врача predicted_target = 1")
            else:
                st.success("Низкая вероятность сердечных заболеваний predicted_target = 0")

    except requests.RequestException as e:
        st.error(f"Запрос не удался: {e}")
