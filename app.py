
# SmartVideoInspector MVP
# Streamlit приложение для анализа видео с поддержкой пользовательских критериев, базового CV и AI-подсказок

import streamlit as st
import cv2
import tempfile
import os
from datetime import datetime
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="SmartVideoInspector", layout="wide")

# Основной заголовок
st.markdown("""
    <style>
    .main-title {
        font-size:36px;
        font-weight:bold;
        color:#1F4E79;
        text-align:center;
        margin-bottom:20px;
    }
    </style>
    <div class=\"main-title\">SmartVideoInspector: AI-анализ видео процессов и ремонта</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

uploaded_video = st.file_uploader("\U0001F4C2 Загрузите видео для анализа:", type=["mp4", "mov"])

st.sidebar.title("\U0001F4DD Настройки анализа")
track_movement = st.sidebar.checkbox("\u2705 Обнаружение задержек в движении", value=True)
movement_threshold = st.sidebar.slider("Порог чувствительности", 10, 100, 30)

# Шаблоны критериев
st.sidebar.markdown("**Выберите шаблон анализа:**")
template = st.sidebar.selectbox("Шаблон", ["По умолчанию", "Контроль остановок", "Анализ сборки", "Обнаружение вращения"])

# Графический выбор зоны
st.sidebar.markdown("**Выделите зону интереса на первом кадре:**")
use_zone = st.sidebar.checkbox("Учитывать зону интереса")
zone_selected = False

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.video(uploaded_video)
    st.info(f"\U0001F3AC Видео содержит {frame_count} кадров при {fps} FPS")

    ret, preview_frame = cap.read()
    if ret:
        st.subheader("Выбор зоны интереса (только для первого кадра)")
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="#00FF00",
            background_image=Image.fromarray(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="rect",
            key="canvas_zone"
        )

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][0]
            x1 = int(obj["left"])
            y1 = int(obj["top"])
            x2 = x1 + int(obj["width"])
            y2 = y1 + int(obj["height"])
            zone_selected = True

    stframe = st.empty()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    prev_gray = None
    log = []
    event_descriptions = []

    with st.spinner("\U0001F52C Анализируем видео..."):
        for i in range(0, frame_count, int(fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            # Зона интереса
            if use_zone and zone_selected:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Анализ движения
            if track_movement:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    roi = diff[y1:y2, x1:x2] if use_zone and zone_selected else diff
                    non_zero_count = np.count_nonzero(roi > 25)
                    if non_zero_count < movement_threshold * 100:
                        ts = str(datetime.now().time())
                        description = f"На кадре {i} обнаружена возможная задержка в зоне."
                        log.append((ts, i, "Задержка движения"))
                        event_descriptions.append(description)
                        cv2.putText(frame, "Задержка!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                prev_gray = gray

            # Визуализация
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

    # Вывод лога
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("\U0001F4C4 Обнаруженные события")
    if log:
        for entry in log:
            st.write(f"[{entry[0]}] Кадр {entry[1]} — {entry[2]}")
    else:
        st.success("\U0001F389 Аномалии не обнаружены")

    # AI-рекомендации
    if event_descriptions:
        st.subheader("\U0001F4A1 AI-рекомендации")
        for desc in event_descriptions:
            st.write(f"➡️ {desc}\n**Совет:** Используйте шаблон '{template}' для дальнейшего анализа. Проверьте узкие места и соответствие нормам.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("\U0001F4BB Посети [gptonline.ai](https://gptonline.ai/) для новых модулей и дополнений!")
