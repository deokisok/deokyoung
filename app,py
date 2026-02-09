# app.py
import datetime as dt
import re
from typing import Dict, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st

# OpenAI (Python SDK v1.x)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # fallback if package missing


# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관 체크인 → 7일 트렌드 → AI 코치 리포트까지 한 번에!")

# =========================
# Sidebar: API Keys
# =========================
with st.sidebar:
    st.header("🔐 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password", placeholder="...")

    st.divider()
    st.caption("💡 키가 없어도 UI/차트/세션 저장은 동작합니다. 리포트/날씨만 제한될 수 있어요.")


# =========================
# Session State Init
# =========================
def _init_state():
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


def _seed_demo_records():
    """6일 샘플 + (오늘 기록은 사용자가 체크인하면 추가)"""
    if st.session_state.demo_inited:
        return

    today = dt.date.today()
    # 최근 6일(오늘 제외) 샘플
    sample = []
    for i in range(6, 0, -1):
        d = today - dt.timedelta(days=i)
        # 가벼운 랜덤 느낌의 패턴(결정적/재현 가능)
        completed = (i * 37) % 6  # 0~5
        mood = max(1, min(10, (i * 19) % 11))  # 1~10
        sample.append(
            {
                "date": d.isoformat(),
                "completed": completed,
                "total": 5,
                "rate": round(completed / 5 * 100, 1),
                "mood": mood,
                "habits": [],  # 데모는 상세 생략
            }
        )
    st.session_state.records = sample
    st.session_state.demo_inited = True


_init_state()
_seed_demo_records()

# =========================
# External APIs
# =========================
def get_weather(city: str, api_key: str) -> Optional[Dict]:
    """
    OpenWeatherMap 현재 날씨
    - 한국어(lang=kr), 섭씨(units=metric)
    - 실패 시 None
    """
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
        data = r.json()
        weather_desc = (data.get("weather") or [{}])[0].get("description")
        temp = (data.get("main") or {}).get("temp")
        feels = (data.get("main") or {}).get("feels_like")
        humidity = (data.get("main") or {}).get("humidity")
        wind = (data.get("wind") or {}).get("speed")
        return {
            "city": city,
            "description": weather_desc,
            "temp_c": temp,
            "feels_like_c": feels,
            "humidity": humidity,
            "wind_mps": wind,
        }
    except Exception:
        return None


def _parse_dog_breed_from_url(url: str) -> str:
    """
    Dog CEO 이미지 URL에서 품종 추출:
    예) https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
        -> hound (afghan)
    """
    try:
        m = re.search(r"/breeds/([^/]+)/", url)
        if not m:
            return "알 수 없음"
        token = m.group(1)  # e.g., "hound-afghan" or "shiba"
        if "-" in token:
            base, sub = token.split("-", 1)
            return f"{base} ({sub})"
        return token
    except Exception:
        return "알 수 없음"


def get_dog_image() -> Optional[Dict]:
    """
    Dog CEO 랜덤 이미지
    - URL과 품종 반환
    - 실패 시 None
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        img_url = data.get("message")
        if not img_url:
            return None
        breed = _parse_dog_breed_from_url(img_url)
        return {"image_url": img_url, "breed": breed}
    except Exception:
        return None


# =========================
# AI Coach Report
# =========================
COACH_SYSTEM_PROMPTS = {
    "스파르타 코치": (
        "너는 매우 엄격하지만 공정한 코치다. 변명은 받아주지 않는다. "
        "짧고 단호하게, 하지만 실행 가능한 구체 지시를 준다."
    ),
    "따뜻한 멘토": (
        "너는 따뜻하고 공감적인 멘토다. 사용자의 감정을 존중하고, "
        "작은 성취를 인정하며 부드럽게 개선 방향을 제시한다."
    ),
    "게임 마스터": (
        "너는 RPG 게임 마스터다. 사용자의 하루를 퀘스트/보상/레벨업 관점으로 해석한다. "
        "유쾌한 톤으로 몰입감 있게 말한다."
    ),
}


def generate_report(
    *,
    openai_key: str,
    coach_style: str,
    city: str,
    weather: Optional[Dict],
    dog: Optional[Dict],
    mood: int,
    checked_habits: List[str],
    unchecked_habits: List[str],
) -> Optional[str]:
    """
    OpenAI에 습관+기분+날씨+강아지 품종 전달 → 지정 형식 리포트 생성
    실패 시 None
    """
    if not openai_key or OpenAI is None:
        return None

    system = COACH_SYSTEM_PROMPTS.get(coach_style, COACH_SYSTEM_PROMPTS["따뜻한 멘토"])

    weather_line = "날씨 정보 없음"
    if weather:
        weather_line = (
            f"{weather.get('city')} / {weather.get('description')} / "
            f"{weather.get('temp_c')}°C(체감 {weather.get('feels_like_c')}°C) / "
            f"습도 {weather.get('humidity')}% / 바람 {weather.get('wind_mps')}m/s"
        )

    dog_line = "강아지 정보 없음"
    if dog:
        dog_line = f"품종: {dog.get('breed')} / 이미지: {dog.get('image_url')}"

    today = dt.date.today().isoformat()

    user_content = f"""
[날짜] {today}
[도시] {city}
[기분] {mood}/10

[오늘 달성 습관]
- {chr(10).join(checked_habits) if checked_habits else "없음"}

[오늘 미달성 습관]
- {chr(10).join(unchecked_habits) if unchecked_habits else "없음"}

[날씨]
- {weather_line}

[오늘의 강아지]
- {dog_line}

요청:
아래 형식을 정확히 지켜서 한국어로 작성해줘. 과도한 장문은 피하고, 실행 가능한 한 줄 액션을 포함해줘.

출력 형식(순서 고정):
1) 컨디션 등급: (S/A/B/C/D 중 하나)
2) 습관 분석: (잘한 점 2개 + 개선점 2개, 불릿)
3) 날씨 코멘트: (한 문단)
4) 내일 미션: (딱 3개, 체크박스 형태로)
5) 오늘의 한마디: (한 줄)
""".strip()

    try:
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else None
    except Exception:
        return None


# =========================
# Habit Check-in UI
# =========================
HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Jeju",
    "Seongnam",
]

st.subheader("✅ 오늘의 체크인")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("#### 습관 체크")
    c1, c2 = st.columns(2, gap="medium")

    # 2열 배치: 3개/2개
    checked = {}
    for idx, (emoji, name) in enumerate(HABITS):
        col = c1 if idx % 2 == 0 else c2
        with col:
            checked[name] = st.checkbox(f"{emoji} {name}", value=False, key=f"habit_{name}")

    st.markdown("#### 기분")
    mood = st.slider("오늘 기분은 어떤가요? (1=최악, 10=최고)", 1, 10, 6)

with right:
    st.markdown("#### 환경 설정")
    city = st.selectbox("도시 선택", CITIES, index=0)
    coach_style = st.radio(
        "코치 스타일",
        ["스파르타 코치", "따뜻한 멘토", "게임 마스터"],
        horizontal=True,
    )

    st.markdown("#### 오늘 한 줄 메모 (선택)")
    note = st.text_area("메모", placeholder="예: 점심 이후 집중이 잘 안 됨 / 저녁 운동 성공!", height=100)

# =========================
# Metrics & Chart Data
# =========================
completed_count = sum(1 for v in checked.values() if v)
rate = round((completed_count / 5) * 100, 1)

m1, m2, m3 = st.columns(3, gap="large")
m1.metric("달성률", f"{rate}%", help="체크한 습관 수 / 5")
m2.metric("달성 습관", f"{completed_count}/5")
m3.metric("기분", f"{mood}/10")

# Save today's record (idempotent per date)
def upsert_today_record():
    today = dt.date.today().isoformat()
    habits_done = [k for k, v in checked.items() if v]
    # replace if exists
    new_rec = {
        "date": today,
        "completed": completed_count,
        "total": 5,
        "rate": rate,
        "mood": mood,
        "habits": habits_done,
        "note": note.strip(),
        "city": city,
        "coach_style": coach_style,
    }
    replaced = False
    for i, r in enumerate(st.session_state.records):
        if r.get("date") == today:
            st.session_state.records[i] = new_rec
            replaced = True
            break
    if not replaced:
        st.session_state.records.append(new_rec)

# always keep today's in session (so chart reflects current UI)
upsert_today_record()

# Build 7-day frame: last 6 days from records + today (sorted, unique by date)
df = pd.DataFrame(st.session_state.records).copy()
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    # keep last 7 days
    df = df.tail(7)
    df["day"] = df["date"].dt.strftime("%m/%d")
else:
    df = pd.DataFrame({"day": [], "rate": [], "mood": []})

st.subheader("📈 최근 7일 트렌드")
if len(df) >= 1:
    chart_df = df[["day", "rate"]].set_index("day")
    st.bar_chart(chart_df, height=260)
else:
    st.info("아직 데이터가 없어요. 체크인하면 7일 차트가 나타납니다.")

# =========================
# Generate Report Button + Results
# =========================
st.divider()
st.subheader("🧠 AI 코치 리포트")

checked_habits = [f"✅ {h}" for _, h in HABITS if checked[h]]
unchecked_habits = [f"⬜ {h}" for _, h in HABITS if not checked[h]]

btn = st.button("컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    with st.spinner("날씨/강아지/AI 코치 리포트를 불러오는 중..."):
        weather = get_weather(city, weather_api_key)
        dog = get_dog_image()

        report = generate_report(
            openai_key=openai_api_key,
            coach_style=coach_style,
            city=city,
            weather=weather,
            dog=dog,
            mood=mood,
            checked_habits=checked_habits,
            unchecked_habits=unchecked_habits,
        )

        st.session_state.last_weather = weather
        st.session_state.last_dog = dog
        st.session_state.last_report = report

# Display cards + report (if available)
weather = st.session_state.last_weather
dog = st.session_state.last_dog
report = st.session_state.last_report

top_left, top_right = st.columns(2, gap="large")

with top_left:
    st.markdown("#### 🌦️ 오늘의 날씨")
    if weather:
        st.success(
            f"**{weather.get('city')}**\n\n"
            f"- {weather.get('description')}\n"
            f"- 기온: {weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C)\n"
            f"- 습도: {weather.get('humidity')}%\n"
            f"- 바람: {weather.get('wind_mps')} m/s"
        )
    else:
        st.warning("날씨 정보를 가져오지 못했어요. OpenWeatherMap API Key와 도시를 확인해 주세요.")

with top_right:
    st.markdown("#### 🐶 오늘의 강아지")
    if dog and dog.get("image_url"):
        st.image(dog["image_url"], use_container_width=True)
        st.caption(f"품종: {dog.get('breed', '알 수 없음')}")
    else:
        st.warning("강아지 이미지를 가져오지 못했어요. 잠시 후 다시 시도해 주세요.")

st.markdown("#### 📝 AI 코치 리포트")
if report:
    st.markdown(report)
else:
    if btn:
        st.error(
            "리포트를 생성하지 못했어요. "
            "OpenAI API Key가 올바른지, openai 패키지가 설치되어 있는지 확인해 주세요."
        )
    else:
        st.info("버튼을 눌러 리포트를 생성하면 여기에 표시됩니다.")

# Share text
st.markdown("#### 📣 공유용 텍스트")
share_lines = [
    f"📊 AI 습관 트래커 ({dt.date.today().isoformat()})",
    f"도시: {city} / 코치: {coach_style}",
    f"달성률: {rate}% ({completed_count}/5) / 기분: {mood}/10",
    "",
    "✅ 달성:",
    *(f"- {h}" for h in [name for _, name in HABITS if checked[name]]) or ["- 없음"],
    "",
    "⬜ 미달성:",
    *(f"- {h}" for h in [name for _, name in HABITS if not checked[name]]) or ["- 없음"],
]
if weather:
    share_lines += [
        "",
        f"🌦️ 날씨: {weather.get('description')} / {weather.get('temp_c')}°C",
    ]
if dog:
    share_lines += [
        f"🐶 오늘의 강아지: {dog.get('breed')}",
    ]
if report:
    share_lines += ["", "🧠 AI 코치 리포트:", report]

st.code("\n".join(share_lines), language="text")

# =========================
# Footer: API Guide
# =========================
with st.expander("🔎 API 안내 / 설정 방법"):
    st.markdown(
        """
**1) OpenAI API Key**
- OpenAI 플랫폼에서 발급한 API 키가 필요합니다.
- 사이드바에 입력하면 `gpt-5-mini` 모델로 리포트를 생성합니다.

**2) OpenWeatherMap API Key**
- OpenWeatherMap에 가입 후 API Key를 발급받아 사이드바에 입력하세요.
- 날씨는 **한국어**(`lang=kr`) + **섭씨**(`units=metric`)로 조회합니다.

**3) 강아지 이미지 (Dog CEO)**
- 무료 공개 API를 사용합니다.
- 네트워크 오류 등으로 실패할 수 있으며, 이 경우 앱은 계속 동작합니다.

**문제 해결 팁**
- 리포트가 안 나오면: OpenAI 키/네트워크/`openai` 패키지 설치 여부를 확인하세요.
- 날씨가 안 나오면: OpenWeatherMap 키가 유효한지 확인하세요.
        """.strip()
    )
