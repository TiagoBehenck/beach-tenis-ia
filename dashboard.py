import streamlit as st
import pandas as pd

from beachtenis_ai import *

st.set_page_config(
    layout="wide"
)

beachTenisIA = BeachTenisIA()
beachTenisIA.run(True)

placeholder = st.empty()

count = 0
status = 'relaxed'

while True:
    frame, landmarks, ts = beachTenisIA.image_q.get()

    if len(landmarks.pose_landmarks) > 0: 
        
        frame, elbow_angle_right = beachTenisIA.find_angle(frame, landmarks, 12, 14, 16, True)
        frame, elbow_angle_left = beachTenisIA.find_angle(frame, landmarks, 11, 13, 15, True)

        result1 = beachTenisIA.analyze_tennis_serve(elbow_angle_right)
        print("Resultado 1:", result1)  

        with placeholder.container():
          col1, col2 = st.columns([0.4, 0.6])
          
          st.image(frame)

        #   col2.markdown("### **Result** " + result1)
          