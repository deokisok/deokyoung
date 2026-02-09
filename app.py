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
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")
st.title("📊 AI 습관 트래커")
st.caption("습관 체크 + 할일(To-do) + 7일 트렌드 + AI 코치 리포트")


# =========================
# Sidebar: API Keys
# =========================
with st.sidebar:
    st.header("🔐 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password")
    st.divider()
    st.caption("API 키가 없어도 체크인/할일/차트는 동작합니다. (리포트/날씨만 제한될 수 있어요)")


# =========================
# Session State
# =========================
def init_state():
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

    # ✅ To-do list
    if "todos" not in st.session_state:
        # each: {"id": str, "text": str, "done": bool, "created": iso}
        st.session_state.todos = []

init_state()


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
        completed = i % 6  # 0~5
        mood = max(1, min(10, i + 3))
        demo.append({"date": d.isoformat(), "completed": completed, "rate": round(completed / 5 * 100, 1), "mood": mood})
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
        params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        return {
            "city": city,
            "desc": j["weather"][0]["description"],
            "temp": j["main"]["temp"],
            "feels": j["main"]["feels_like"],
            "humidity": j["main"].get("humidity"),
            "wind": (j.get("wind") or {}).get("speed"),
        }
    except Exception:
        return None


def get_dog_image() -> Optional[Dict]:
    try:
        r = requests.get("https://dog.ceo/api/breeds/image/random", timeout=10)
        if r.status_code != 200:
            return None
        url = r.json().get("message")
        if not url:
            return None
        m = re.search(r"/breeds/([^/]+)/", url)
        breed = m.group(1).replace("-", " ") if m else "알 수 없음"
        return {"url": url, "breed": breed}
    except Exception:
        return None


# =========================
# AI Report
# =========================
SYSTEM_PROMPTS = {
    "스파르타 코치": "너는 엄격한 코치다. 직설적이고 실행 중심으로 말한다. 변명은 컷.",
    "따뜻한 멘토": "너는 따뜻하고 공감적인 멘토다. 칭찬과 개선 방향을 부드럽게 제시한다.",
    "게임 마스터": "너는 RPG 게임 마스터다. 퀘스트/보상/레벨업 관점으로 유쾌하게 말한다.",
}


def generate_report(
    openai_key: str,
    style: str,
    habits_done: List[str],
    habits_todo: List[str],
    mood: int,
    city: str,
    weather: Optional[Dict],
    dog: Optional[Dict],
    todos_done: List[str],
    todos_todo: List[str],
) -> Optional[str]:
    if not openai_key or OpenAI is None:
        return None

    weather_text = (
        f"{weather['city']} / {weather['desc']} / {weather['temp']}°C(체감 {weather['feels']}°C)"
        if weather
        else "날씨 정보 없음"
    )
    dog_text = dog["breed"] if dog else "강아지 없음"

    today = dt.date.today().isoformat()

    user_prompt = f"""
[날짜] {today}
[도시] {city}
[기분] {mood}/10

[습관 - 완료]
{habits_done if habits_done else ["없음"]}

[습관 - 미완료]
{habits_todo if habits_todo else ["없음"]}

[할일 - 완료]
{todos_done if todos_done else ["없음"]}

[할일 - 미완료]
{todos_todo if todos_todo else ["없음"]}

[날씨]
{weather_text}

[오늘의 강아지]
{dog_text}

요청:
아래 형식을 정확히 지켜 한국어로 작성해줘. 너무 길게 쓰지 말고 실행 가능한 액션을 포함해줘.

출력 형식(순서 고정):
1) 컨디션 등급: (S/A/B/C/D 중 하나)
2) 습관 분석: (잘한 점 2개 + 개선점 2개, 불릿)
3) 할일 코멘트: (할일 진행 상황 + 내일 우선순위 제안, 불릿 3개)
4) 날씨 코멘트: (한 문단)
5) 내일 미션: (딱 3개, 체크박스 형태로)
6) 오늘의 한마디: (한 줄)
""".strip()

    try:
        client = OpenAI(api_key=openai_key)
        res = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["따뜻한 멘토"])},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        text = (res.choices[0].message.content or "").strip()
        return text or None
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

CITIES = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon", "Gwangju", "Ulsan", "Suwon", "Jeju", "Seongnam"]

st.subheader("✅ 오늘의 체크인")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("#### 습관 체크")
    col1, col2 = st.columns(2, gap="medium")
    checked = {}

    for i, (emo, name) in enumerate(HABITS):
        col = col1 if i % 2 == 0 else col2
        with col:
            checked[name] = st.checkbox(f"{emo} {name}", key=f"habit_{name}")

    st.markdown("#### 기분")
    mood = st.slider("오늘 기분은 어떤가요? (1=최악, 10=최고)", 1, 10, 6, key="mood")

with right:
    st.markdown("#### 환경 설정")
    city = st.selectbox("도시", CITIES, index=0, key="city")
    coach_style = st.radio("코치 스타일", list(SYSTEM_PROMPTS.keys()), horizontal=True, key="coach_style")

    st.markdown("#### 오늘 한 줄 메모 (선택)")
    note = st.text_area("메모", placeholder="예: 점심 이후 집중이 잘 안 됨 / 저녁 운동 성공!", height=90, key="note")


# =========================
# ✅ To-do List UI (추가)
# =========================
st.subheader("🧾 할일 리스트 (To-do)")

todo_left, todo_right = st.columns([1.2, 0.8], gap="large")

with todo_left:
    st.markdown("#### ➕ 할일 추가")
    with st.form("todo_add_form", clear_on_submit=True):
        new_todo = st.text_input("할일", placeholder="예: 영어 단어 30개 / 이력서 수정 / 산책 20분")
        add = st.form_submit_button("추가", use_container_width=True)
        if add:
            text = (new_todo or "").strip()
            if text:
                todo_id = f"{dt.datetime.now().timestamp():.6f}"
                st.session_state.todos.append({"id": todo_id, "text": text, "done": False, "created": dt.datetime.now().isoformat()})
            else:
                st.warning("할일 내용을 입력해 주세요.")

with todo_right:
    st.markdown("#### 🧹 관리")
    cA, cB = st.columns(2)
    with cA:
        if st.button("완료 항목 삭제", use_container_width=True):
            st.session_state.todos = [t for t in st.session_state.todos if not t.get("done", False)]
    with cB:
        if st.button("전체 삭제", use_container_width=True):
            st.session_state.todos = []

st.markdown("#### ✅ 오늘의 할일")
if not st.session_state.todos:
    st.info("아직 할일이 없어요. 오른쪽에서 추가해 주세요.")
else:
    # 체크박스 렌더링
    for t in st.session_state.todos:
        key = f"todo_done_{t['id']}"
        # 현재 상태를 위젯 기본값으로 반영
        current = st.checkbox(f"🗒️ {t['text']}", value=bool(t.get("done", False)), key=key)
        t["done"] = current  # 상태 반영


# =========================
# Metrics
# =========================
completed = sum(1 for v in checked.values() if v)
rate = round(completed / 5 * 100, 1)

todo_done_cnt = sum(1 for t in st.session_state.todos if t.get("done"))
todo_total_cnt = len(st.session_state.todos)
todo_rate = round((todo_done_cnt / todo_total_cnt) * 100, 1) if todo_total_cnt else 0.0

m1, m2, m3 = st.columns(3, gap="large")
m1.metric("달성률(습관)", f"{rate}%")
m2.metric("달성 습관", f"{completed}/5")
m3.metric("기분", f"{mood}/10")

# 보너스: 할일 지표
st.caption(f"🧾 할일 진행률: **{todo_rate}%**  ({todo_done_cnt}/{todo_total_cnt})")


# =========================
# Save Today (idempotent)
# =========================
today = dt.date.today().isoformat()
today_record = {
    "date": today,
    "completed": completed,
    "rate": rate,
    "mood": mood,
    "note": (note or "").strip(),
    "city": city,
    "coach_style": coach_style,
    "todo_total": todo_total_cnt,
    "todo_done": todo_done_cnt,
}

# upsert
replaced = False
for i, r in enumerate(st.session_state.records):
    if r.get("date") == today:
        st.session_state.records[i] = today_record
        replaced = True
        break
if not replaced:
    st.session_state.records.append(today_record)


# =========================
# 7-day Chart
# =========================
st.subheader("📈 최근 7일 달성률(습관)")
df = pd.DataFrame(st.session_state.records).copy()
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").tail(7)
    df["day"] = df["date"].dt.strftime("%m/%d")
    st.bar_chart(df.set_index("day")["rate"], height=260)
else:
    st.info("아직 데이터가 없어요.")


# =========================
# AI Report
# =========================
st.divider()
st.subheader("🧠 AI 코치 리포트")

if st.button("컨디션 리포트 생성", type="primary", use_container_width=True):
    with st.spinner("날씨/강아지/AI 리포트를 불러오는 중..."):
        weather = get_weather(city, weather_api_key)
        dog = get_dog_image()

        habits_done = [name for _, name in HABITS if checked[name]]
        habits_todo = [name for _, name in HABITS if not checked[name]]

        todos_done = [t["text"] for t in st.session_state.todos if t.get("done")]
        todos_todo = [t["text"] for t in st.session_state.todos if not t.get("done")]

        report = generate_report(
            openai_key=openai_api_key,
            style=coach_style,
            habits_done=habits_done,
            habits_todo=habits_todo,
            mood=mood,
            city=city,
            weather=weather,
            dog=dog,
            todos_done=todos_done,
            todos_todo=todos_todo,
        )

        st.session_state.last_weather = weather
        st.session_state.last_dog = dog
        st.session_state.last_report = report


# =========================
# Results
# =========================
w_col, d_col = st.columns(2, gap="large")

with w_col:
    st.markdown("### 🌦️ 날씨")
    weather = st.session_state.last_weather
    if weather:
        st.success(
            f"**{weather.get('city')}**\n\n"
            f"- {weather.get('desc')}\n"
            f"- 기온: {weather.get('temp')}°C (체감 {weather.get('feels')}°C)\n"
            + (f"- 습도: {weather.get('humidity')}%\n" if weather.get("humidity") is not None else "")
            + (f"- 바람: {weather.get('wind')} m/s\n" if weather.get("wind") is not None else "")
        )
    else:
        st.info("날씨 정보 없음 (OpenWeatherMap API Key 필요)")

with d_col:
    st.markdown("### 🐶 오늘의 강아지")
    dog = st.session_state.last_dog
    if dog and dog.get("url"):
        st.image(dog["url"], use_container_width=True)
        st.caption(f"품종: {dog.get('breed', '알 수 없음')}")
    else:
        st.info("강아지 없음 (네트워크 상황에 따라 실패할 수 있어요)")

st.markdown("### 📝 AI 리포트")
if st.session_state.last_report:
    st.markdown(st.session_state.last_report)
else:
    st.info("버튼을 눌러 리포트를 생성하면 여기에 표시됩니다. (OpenAI API Key 필요)")


# =========================
# Share Text
# =========================
st.subheader("📣 공유용 텍스트")

done_habits_lines = [f"- {name}" for _, name in HABITS if checked[name]]
todo_habits_lines = [f"- {name}" for _, name in HABITS if not checked[name]]

done_todos_lines = [f"- {t['text']}" for t in st.session_state.todos if t.get("done")]
todo_todos_lines = [f"- {t['text']}" for t in st.session_state.todos if not t.get("done")]

share_lines = [
    f"📊 AI 습관 트래커 ({today})",
    f"도시: {city} / 코치: {coach_style}",
    f"습관 달성률: {rate}% ({completed}/5) / 기분: {mood}/10",
    f"할일 진행률: {todo_rate}% ({todo_done_cnt}/{todo_total_cnt})",
    "",
    "✅ 습관 달성:",
    *(done_habits_lines if done_habits_lines else ["- 없음"]),
    "",
    "⬜ 습관 미달성:",
    *(todo_habits_lines if todo_habits_lines else ["- 없음"]),
    "",
    "🧾 할일 완료:",
    *(done_todos_lines if done_todos_lines else ["- 없음"]),
    "",
    "🗒️ 할일 미완료:",
    *(todo_todos_lines if todo_todos_lines else ["- 없음"]),
]

weather = st.session_state.last_weather
dog = st.session_state.last_dog
report = st.session_state.last_report

if weather:
    share_lines += ["", f"🌦️ 날씨: {weather.get('desc')} / {weather.get('temp')}°C"]
if dog:
    share_lines += [f"🐶 오늘의 강아지: {dog.get('breed')}"]
if report:
    share_lines += ["", "🧠 AI 코치 리포트:", report]

st.code("\n".join(share_lines), language="text")


# =========================
# Footer
# =========================
with st.expander("ℹ️ API 안내"):
    st.markdown(
        """
- **OpenAI**: AI 코치 리포트 생성 (`gpt-5-mini`)  
- **OpenWeatherMap**: 현재 날씨(한국어, 섭씨)  
- **Dog CEO**: 랜덤 강아지 이미지/품종  

API 호출 실패 시에도 앱은 계속 동작합니다.
"""
    )
