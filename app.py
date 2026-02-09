# app.py
import datetime as dt
import re
from typing import Optional, Dict, List

import pandas as pd
import requests
import streamlit as st

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="AI 습관 트래커",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI 습관 트래커")
st.caption("습관 체크 → 7일 트렌드 → AI 코치 리포트")

# =========================
# Sidebar: API Keys
# =========================
with st.sidebar:
    st.header("🔐 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password")

    st.divider()
    st.caption("API 키가 없어도 체크인과 차트는 정상 동작합니다.")


# =========================
# Session State
# =========================
if "records" not in st.session_state:
    st.session_state.records = []

if "demo_inited" not in st.session_state:
    st.session_state.demo_inited = False

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "last_weather" not in st.session_state:
    st.session_state.last_weather = None

if "last_dog" not in st.session_state:
    st.session_state.last_dog = None


# =========================
# Demo Data (6 days)
# =========================
def seed_demo():
    if st.session_state.demo_inited:
        return

    today = dt.date.today()
    demo = []
    for i in range(6, 0, -1):
        d = today - dt.timedelta(days=i)
        completed = i % 6
        mood = max(1, min(10, i + 3))
        demo.append(
            {
                "date": d.isoformat(),
                "completed": completed,
                "rate": round(completed / 5 * 100, 1),
                "mood": mood,
            }
        )
    st.session_state.records = demo
    st.session_state.demo_inited = True


seed_demo()


# =========================
# External APIs
# =========================
def get_weather(city: str, api_key: str) -> Optional[Dict]:
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "kr",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        return {
            "city": city,
            "desc": j["weather"][0]["description"],
            "temp": j["main"]["temp"],
            "feels": j["main"]["feels_like"],
        }
    except Exception:
        return None


def get_dog_image() -> Optional[Dict]:
    try:
        r = requests.get("https://dog.ceo/api/breeds/image/random", timeout=10)
        if r.status_code != 200:
            return None
        url = r.json()["message"]
        m = re.search(r"/breeds/([^/]+)/", url)
        breed = m.group(1).replace("-", " ") if m else "알 수 없음"
        return {"url": url, "breed": breed}
    except Exception:
        return None


# =========================
# AI Report
# =========================
SYSTEM_PROMPTS = {
    "스파르타 코치": "너는 엄격한 코치다. 직설적이고 실행 중심으로 말한다.",
    "따뜻한 멘토": "너는 따뜻하고 공감적인 멘토다.",
    "게임 마스터": "너는 RPG 게임 마스터다. 퀘스트처럼 말한다.",
}


def generate_report(
    openai_key: str,
    style: str,
    habits_done: List[str],
    habits_todo: List[str],
    mood: int,
    weather: Optional[Dict],
    dog: Optional[Dict],
) -> Optional[str]:
    if not openai_key or OpenAI is None:
        return None

    weather_text = (
        f"{weather['desc']} / {weather['temp']}°C"
        if weather
        else "날씨 정보 없음"
    )
    dog_text = dog["breed"] if dog else "강아지 없음"

    user_prompt = f"""
기분: {mood}/10
완료 습관: {habits_done}
미완료 습관: {habits_todo}
날씨: {weather_text}
강아지: {dog_text}

출력 형식:
1) 컨디션 등급(S~D)
2) 습관 분석 (불릿)
3) 날씨 코멘트
4) 내일 미션 3개 (체크박스)
5) 오늘의 한마디
"""

    try:
        client = OpenAI(api_key=openai_key)
        res = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[style]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return res.choices[0].message.content
    except Exception:
        return None


# =========================
# Habit UI
# =========================
HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

CITIES = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Ulsan", "Suwon", "Jeju", "Seongnam",
]

st.subheader("✅ 오늘의 체크인")

c1, c2 = st.columns(2)
checked = {}

for i, (emo, name) in enumerate(HABITS):
    col = c1 if i % 2 == 0 else c2
    with col:
        checked[name] = st.checkbox(f"{emo} {name}")

mood = st.slider("기분", 1, 10, 6)
city = st.selectbox("도시", CITIES)
coach_style = st.radio("코치 스타일", list(SYSTEM_PROMPTS.keys()), horizontal=True)

# =========================
# Metrics
# =========================
completed = sum(checked.values())
rate = round(completed / 5 * 100, 1)

m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{rate}%")
m2.metric("완료 습관", f"{completed}/5")
m3.metric("기분", f"{mood}/10")

# =========================
# Save Today
# =========================
today = dt.date.today().isoformat()
st.session_state.records.append(
    {"date": today, "completed": completed, "rate": rate, "mood": mood}
)

df = pd.DataFrame(st.session_state.records).drop_duplicates("date").tail(7)
df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.strftime("%m/%d")

st.subheader("📈 7일 달성률")
st.bar_chart(df.set_index("day")["rate"])

# =========================
# Report Button
# =========================
st.subheader("🧠 AI 코치 리포트")

if st.button("컨디션 리포트 생성", type="primary"):
    weather = get_weather(city, weather_api_key)
    dog = get_dog_image()

    done_list = [name for name in checked if checked[name]]
    todo_list = [name for name in checked if not checked[name]]

    report = generate_report(
        openai_api_key,
        coach_style,
        done_list,
        todo_list,
        mood,
        weather,
        dog,
    )

    st.session_state.last_weather = weather
    st.session_state.last_dog = dog
    st.session_state.last_report = report

# =========================
# Results
# =========================
col_w, col_d = st.columns(2)

with col_w:
    st.markdown("### 🌦️ 날씨")
    if st.session_state.last_weather:
        w = st.session_state.last_weather
        st.success(f"{w['desc']} / {w['temp']}°C")
    else:
        st.info("날씨 정보 없음")

with col_d:
    st.markdown("### 🐶 오늘의 강아지")
    if st.session_state.last_dog:
        st.image(st.session_state.last_dog["url"])
        st.caption(st.session_state.last_dog["breed"])
    else:
        st.info("강아지 없음")

st.markdown("### 📝 AI 리포트")
if st.session_state.last_report:
    st.markdown(st.session_state.last_report)
else:
    st.info("아직 리포트가 없습니다.")

# =========================
# Share Text (FIXED)
# =========================
done_list = [f"- {name}" for name in checked if checked[name]]
todo_list = [f"- {name}" for name in checked if not checked[name]]

share_text = [
    f"📊 AI 습관 트래커 ({today})",
    f"달성률: {rate}% / 기분: {mood}/10",
    "",
    "✅ 달성:",
    *(done_list if done_list else ["- 없음"]),
    "",
    "⬜ 미달성:",
    *(todo_list if todo_list else ["- 없음"]),
]

st.subheader("📣 공유용 텍스트")
st.code("\n".join(share_text))

# =========================
# Footer
# =========================
with st.expander("ℹ️ API 안내"):
    st.markdown(
        """
- OpenAI: AI 코치 리포트 생성  
- OpenWeatherMap: 현재 날씨  
- Dog CEO: 랜덤 강아지 이미지  

API 실패 시에도 앱은 계속 동작합니다.
"""
    )
