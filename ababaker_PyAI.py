"""
UTM Traffic Monitoring System
Advanced Vehicle Detection, Tracking, Speed Estimation & AI Analysis
Created by ABABAKER NAZAR
OPTIMIZED VERSION - 3-5x FASTER
"""

import streamlit as st
import cv2
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from io import BytesIO

st.set_page_config(page_title="UTM Traffic Monitoring System", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
    
    /* Main Background - Cream */
    .main {
        background-color: #FAF7F2;
    }
    
    .block-container {
        background-color: #FAF7F2;
    }
    
    .main-header {
        font-size: 3rem; 
        font-weight: 900; 
        background: linear-gradient(135deg, #FAF7F2 0%, #FFFBF5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 15px rgba(250, 247, 242, 0.4);
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(250, 247, 242, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.8)); }
    }
    
    /* 3D Card Effects - Reversed */
    .stMetric {
        background: linear-gradient(145deg, #A00000, #6B0000);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 
            5px 5px 15px rgba(0, 0, 0, 0.4),
            -5px -5px 15px rgba(255, 255, 255, 0.1),
            inset 2px 2px 5px rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stMetric:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 
            8px 8px 20px rgba(0, 0, 0, 0.6),
            -8px -8px 20px rgba(255, 255, 255, 0.2);
    }
    
    /* Enhanced Alert Boxes - Reversed */
    .alert-box {
        background: #FAF7F2;
        color: #8B0000;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        font-weight: bold;
        box-shadow: 
            0 8px 20px rgba(0, 0, 0, 0.4),
            inset 0 -2px 10px rgba(139,0,0,0.2);
        border-left: 5px solid #FFFBF5;
        animation: pulse-alert 2s infinite;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
    }
    
    @keyframes pulse-alert {
        0%, 100% { box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4); }
        50% { box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7); }
    }
    
    .warning-box {
        background: #E8DCC6;
        color: #8B0000;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        font-weight: bold;
        box-shadow: 
            0 8px 20px rgba(0, 0, 0, 0.3),
            inset 0 -2px 10px rgba(139,0,0,0.2);
        border-left: 5px solid #FAF7F2;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #8B0000 0%, #A00000 100%);
        color: #FAF7F2;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        font-weight: bold;
        box-shadow: 
            0 8px 20px rgba(0, 0, 0, 0.3),
            inset 0 -2px 10px rgba(255,255,255,0.1);
        border-left: 5px solid #4caf50;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
    }
    
    /* Reversed Sidebar */
    [data-testid="stSidebar"] {
        background: #FAF7F2 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #8B0000 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #8B0000 !important;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] p {
        color: #8B0000 !important;
    }
    
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #8B0000 !important;
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: #8B0000 !important;
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #FAF7F2 !important;
    }
    
    [data-testid="stSidebar"] input[type="range"]::-webkit-slider-thumb {
        background-color: #8B0000 !important;
        border: 2px solid #FAF7F2 !important;
    }
    
    [data-testid="stSidebar"] input {
        background-color: rgba(139, 0, 0, 0.1) !important;
        color: #8B0000 !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #8B0000 !important;
        transition: all 0.3s ease;
        padding: 10px;
        border-radius: 8px;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(139, 0, 0, 0.2);
        transform: translateX(5px);
        font-weight: 700;
    }
    
    /* Tab Styling - Hover Effects */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #FFFBF5;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 15px rgba(139, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(139,0,0,0.08);
        border-radius: 8px;
        color: #8B0000;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(139,0,0,0.2);
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 4px 12px rgba(139, 0, 0, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: #8B0000;
        color: #FAF7F2;
        box-shadow: 0 6px 20px rgba(139,0,0,0.8);
        transform: translateY(-3px);
        font-size: 1.05em;
    }
    
    .stButton > button {
        background: #FAF7F2;
        color: #8B0000;
        border: 2px solid #8B0000;
        border-radius: 10px;
        padding: 12px 24px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.05);
        background: #8B0000;
        color: #FAF7F2;
        box-shadow: 0 6px 25px rgba(139, 0, 0, 0.8);
        letter-spacing: 1px;
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(1.02);
    }
    
    /* Progress Bar - Reversed */
    .stProgress > div > div > div {
        background: #FAF7F2;
        border-radius: 10px;
    }
    
    /* Footer with Gradient */
    .footer {
        text-align: center;
        font-size: 0.9rem;
        background: linear-gradient(135deg, #FAF7F2 0%, #FFFBF5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 40px;
        padding-top: 30px;
        border-top: 3px solid;
        border-image: linear-gradient(90deg, #FAF7F2 0%, #FFFBF5 100%) 1;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }
    
    [data-testid="stVideo"] {
        border-radius: 15px;
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(250, 247, 242, 0.2);
        overflow: hidden;
    }
    
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        background-color: #A00000;
        color: #FAF7F2;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        color: #FAF7F2 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #FAF7F2 0%, #FFFBF5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stPlotlyChart {
        background: #A00000;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 
            5px 5px 15px rgba(0, 0, 0, 0.3),
            -5px -5px 15px rgba(255, 255, 255, 0.1);
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid rgba(139, 0, 0, 0.3);
        transition: all 0.3s ease;
        background-color: #FAF7F2;
        color: #8B0000;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #FAF7F2;
        box-shadow: 0 0 0 3px rgba(250, 247, 242, 0.3);
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(250, 247, 242, 0.1) 0%, rgba(255, 251, 245, 0.1) 100%);
        border-radius: 10px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        color: #FAF7F2;
    }
    
    .stSuccess {
        background: #A00000;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        color: #FAF7F2;
    }
    
    .stError {
        background: #FAF7F2;
        border-radius: 10px;
        border-left: 5px solid #DC143C;
        color: #8B0000;
    }
    
    .stInfo {
        background: #A00000;
        color: #FAF7F2;
        border-radius: 10px;
        border-left: 5px solid #FFFBF5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="position: relative; width: 100%; height: 800px; margin-bottom: 30px; border-radius: 15px; overflow: hidden; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);">
    <video autoplay muted loop playsinline style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 1;" onended="this.play()">
        <source src="https://raw.githubusercontent.com/ababakerosman0001/UTM-Aerial-View/main/VID_20251012_145221_851.mp4" type="video/mp4">
    </video>
    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.2); z-index: 2; display: flex; flex-direction: column; justify-content: center; align-items: center;">
        <h1 style="color: #FAF7F2; font-size: 3.5rem; font-weight: 900; font-family: 'Orbitron', sans-serif; letter-spacing: 3px; margin: 0; text-shadow: 0 4px 20px rgba(0, 0, 0, 0.8); z-index: 3; text-align: center; animation: glow 2s ease-in-out infinite alternate;">UTM TRAFFIC MONITORING</h1>
        <p style="color: #FAF7F2; font-size: 1.4rem; font-family: 'Rajdhani', sans-serif; font-weight: 500; letter-spacing: 1px; margin-top: 15px; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.8); z-index: 3; text-align: center;">Advanced Vehicle Detection • Dual-Line Speed Detection • AI Analysis</p>
    </div>
</div>
<script>
    var video = document.querySelector('video');
    if (video) {
        video.addEventListener('ended', function() {
            this.currentTime = 0;
            this.play();
        }, false);
    }
</script>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_gemini():
    try:
        from google import genai
        return genai.Client(api_key=)
    except:
        return None

@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("./best.pt")
    except:
        st.error("Model loading failed")
        return None

gemini_client = initialize_gemini()
model = load_yolo_model()

if 'events_log' not in st.session_state:
    st.session_state.events_log = []
if 'heatmap_data' not in st.session_state:
    st.session_state.heatmap_data = None
if 'cars_entered' not in st.session_state:
    st.session_state.cars_entered = 0
if 'cars_exited' not in st.session_state:
    st.session_state.cars_exited = 0
if 'crossed_objects' not in st.session_state:
    st.session_state.crossed_objects = set()
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'speed_violations' not in st.session_state:
    st.session_state.speed_violations = []
if 'parking_vehicles' not in st.session_state:
    st.session_state.parking_vehicles = set()
if 'email_logs' not in st.session_state:
    st.session_state.email_logs = []

def send_email_alert(subject, message, recipient, sender, api_key):
    try:
        import requests
        sender = sender.strip().lower()
        recipient = recipient.strip().lower()
        api_key = api_key.strip()
        
        if not sender or not api_key or not recipient:
            raise ValueError("Missing email credentials")
        
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "personalizations": [{"to": [{"email": recipient}]}],
            "from": {"email": sender},
            "subject": subject,
            "content": [{"type": "text/html", "value": message}]
        }
        
        response = requests.post("https://api.sendgrid.com/v3/mail/send", json=data, headers=headers, timeout=20)
        
        if response.status_code in [200, 202]:
            log_msg = f"Email sent to {recipient[:15]}... at {datetime.now().strftime('%H:%M:%S')}"
            st.session_state.email_logs.append(log_msg)
            return True
        else:
            st.session_state.email_logs.append(f"SendGrid error {response.status_code}")
            return False
    except Exception as e:
        st.session_state.email_logs.append(f"Error: {str(e)[:50]}")
        return False

def log_alert_to_sheet(alert_type, details, vehicle_id="", speed="", duration=""):
    """Log alerts to Google Sheet"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        sheet_id = "1R2Fv8_SwIcz4f90qV3i3qaV6FRaN8jRoyAjL11xst1E"
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Check if credentials file exists
        if not Path('credentials.json').exists():
            st.warning("credentials.json not found. Sheet logging disabled.")
            return False
        
        creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).sheet1
        
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            alert_type,
            str(vehicle_id),
            details,
            str(speed) if speed else "",
            str(duration) if duration else ""
        ]
        
        sheet.append_row(row)
        st.session_state.email_logs.append(f"✓ Logged to sheet: {alert_type}")
        return True
        
    except Exception as e:
        st.session_state.email_logs.append(f"Sheet error: {str(e)[:50]}")
        return False

class EventLogger:
    def __init__(self, filename="events_log.csv"):
        self.filename = filename
        self.columns = ['timestamp', 'event_type', 'object_class', 'object_id', 'description', 'duration', 'confidence', 'speed']
        self.buffer = []
        self.buffer_size = 10
        
        if not Path(filename).exists():
            pd.DataFrame(columns=self.columns).to_csv(filename, index=False)
    
    def log_event(self, event_type, object_class, object_id, description, duration=0, confidence=0, speed=0):
        event = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'event_type': event_type,
            'object_class': object_class,
            'object_id': object_id,
            'description': description,
            'duration': duration,
            'confidence': confidence,
            'speed': speed
        }
        self.buffer.append(event)
        st.session_state.events_log.append(event)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
        return event
    
    def flush(self):
        if self.buffer:
            try:
                pd.DataFrame(self.buffer).to_csv(self.filename, mode='a', header=False, index=False)
                self.buffer = []
            except:
                pass

logger = EventLogger()

class AdvancedTracker:
    def __init__(self, max_disappeared=30, line_position=0.5, line_orientation="horizontal", pixels_per_meter=10, line2_position=None):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.object_history = {}
        self.object_states = {}
        self.line_position = line_position
        self.line2_position = line2_position
        self.line_orientation = line_orientation
        self.pixels_per_meter = pixels_per_meter
        self.line_crossing_times = {}
    
    def register(self, centroid, class_name, confidence, frame_height, frame_width):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.object_history[self.next_object_id] = [centroid]
        
        if self.line_orientation == "horizontal":
            line_y = frame_height * self.line_position
            entry_side = 'top' if centroid[1] < line_y else 'bottom'
        else:
            line_x = frame_width * self.line_position
            entry_side = 'left' if centroid[0] < line_x else 'right'
        
        self.object_states[self.next_object_id] = {
            'class': class_name,
            'first_seen': time.time(),
            'is_stopped': False,
            'stop_start_time': None,
            'confidence': confidence,
            'crossed_line1': False,
            'crossed_line2': False,
            'entry_side': entry_side,
            'speeds': [],
            'avg_speed': 0
        }
        logger.log_event('vehicle_detected', class_name, self.next_object_id, f"{class_name} detected", confidence=confidence)
        self.next_object_id += 1
    
    def deregister(self, object_id):
        # Don't log when vehicle leaves frame - removed "disappeared" event
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.object_states:
            del self.object_states[object_id]
        if object_id in self.object_history:
            del self.object_history[object_id]
        if object_id in self.line_crossing_times:
            del self.line_crossing_times[object_id]
    
    def estimate_speed_from_lines(self, object_id, distance_meters):
        if object_id not in self.line_crossing_times:
            return 0
        times = self.line_crossing_times[object_id]
        if len(times) < 2:
            return 0
        time_diff = times[-1] - times[-2]
        if time_diff <= 0:
            return 0
        speed_kmh = (distance_meters / time_diff) * 3.6
        return speed_kmh
    
    def estimate_speed(self, object_id):
        if object_id not in self.object_history or len(self.object_history[object_id]) < 5:
            return 0
        recent = self.object_history[object_id][-5:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        distance = np.sqrt(dx**2 + dy**2) / self.pixels_per_meter
        speed = distance * 3.6
        self.object_states[object_id]['speeds'].append(speed)
        if len(self.object_states[object_id]['speeds']) > 10:
            self.object_states[object_id]['speeds'].pop(0)
        self.object_states[object_id]['avg_speed'] = np.mean(self.object_states[object_id]['speeds']) if self.object_states[object_id]['speeds'] else 0
        return self.object_states[object_id]['avg_speed']
    
    def check_line_crossing(self, object_id, current_pos, frame_height, frame_width, distance_meters=None, speed_limit=60, email_config=None):
        if object_id not in self.object_states:
            return None
        
        entry_side = self.object_states[object_id]['entry_side']
        result = None
        
        if self.line_orientation == "horizontal":
            line_y = frame_height * self.line_position
            
            if not self.object_states[object_id]['crossed_line1']:
                if (entry_side == 'top' and current_pos[1] > line_y) or (entry_side == 'bottom' and current_pos[1] < line_y):
                    self.object_states[object_id]['crossed_line1'] = True
                    self.line_crossing_times[object_id] = [time.time()]
                    st.session_state.cars_entered += 1
                    logger.log_event('vehicle_entered', self.object_states[object_id]['class'], object_id, "Vehicle entered zone", 0, self.object_states[object_id]['confidence'])
                    result = 'crossed_line1'
            
            if self.line2_position and not self.object_states[object_id]['crossed_line2'] and self.object_states[object_id]['crossed_line1']:
                line2_y = frame_height * self.line2_position
                
                if (entry_side == 'top' and current_pos[1] > line2_y) or (entry_side == 'bottom' and current_pos[1] < line2_y):
                    self.object_states[object_id]['crossed_line2'] = True
                    self.line_crossing_times[object_id].append(time.time())
                    st.session_state.cars_exited += 1
                    
                    speed = self.estimate_speed_from_lines(object_id, distance_meters) if distance_meters else 0
                    self.object_states[object_id]['avg_speed'] = speed
                    logger.log_event('vehicle_exited', self.object_states[object_id]['class'], object_id, f"Vehicle exited - Speed: {speed:.1f}km/h", 0, self.object_states[object_id]['confidence'], speed)
                    result = 'crossed_line2'
                    
                    if speed > speed_limit and speed > 0:
                        st.session_state.speed_violations.append({'id': object_id, 'speed': speed})
                        alert_msg = f"SPEED: ID:{object_id} {speed:.1f}km/h (limit {speed_limit})"
                        st.session_state.alerts.append(alert_msg)
                        
                        # LOG TO GOOGLE SHEET
                        log_alert_to_sheet(
                            alert_type="Speed Violation",
                            details=f"Exceeded limit by {speed - speed_limit:.1f} km/h",
                            vehicle_id=object_id,
                            speed=f"{speed:.1f}",
                            duration=""
                        )
                        
                        if email_config and email_config['enabled'] and email_config['sender'] and email_config['api_key']:
                            subject = f"Speed Violation: {speed:.1f}km/h"
                            message = f"""<html><body style="font-family: Arial, sans-serif;">
                            <h3 style="color: #DC143C;">Speed Violation Alert</h3>
                            <p><b>Vehicle ID:</b> {object_id}</p>
                            <p><b>Speed:</b> {speed:.1f} km/h</p>
                            <p><b>Speed Limit:</b> {speed_limit} km/h</p>
                            <p><b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                            </body></html>"""
                            send_email_alert(subject, message, email_config['recipient'], email_config['sender'], email_config['api_key'])
        
        elif self.line_orientation == "vertical":
            line_x = frame_width * self.line_position
            
            if not self.object_states[object_id]['crossed_line1']:
                if (entry_side == 'left' and current_pos[0] > line_x) or (entry_side == 'right' and current_pos[0] < line_x):
                    self.object_states[object_id]['crossed_line1'] = True
                    self.line_crossing_times[object_id] = [time.time()]
                    st.session_state.cars_entered += 1
                    logger.log_event('vehicle_entered', self.object_states[object_id]['class'], object_id, "Vehicle entered zone", 0, self.object_states[object_id]['confidence'])
                    result = 'crossed_line1'
            
            if self.line2_position and not self.object_states[object_id]['crossed_line2'] and self.object_states[object_id]['crossed_line1']:
                line2_x = frame_width * self.line2_position
                
                if (entry_side == 'left' and current_pos[0] > line2_x) or (entry_side == 'right' and current_pos[0] < line2_x):
                    self.object_states[object_id]['crossed_line2'] = True
                    self.line_crossing_times[object_id].append(time.time())
                    st.session_state.cars_exited += 1
                    
                    speed = self.estimate_speed_from_lines(object_id, distance_meters) if distance_meters else 0
                    self.object_states[object_id]['avg_speed'] = speed
                    logger.log_event('vehicle_exited', self.object_states[object_id]['class'], object_id, f"Vehicle exited - Speed: {speed:.1f}km/h", 0, self.object_states[object_id]['confidence'], speed)
                    result = 'crossed_line2'
                    
                    if speed > speed_limit and speed > 0:
                        st.session_state.speed_violations.append({'id': object_id, 'speed': speed})
                        st.session_state.alerts.append(f"SPEED: ID:{object_id} {speed:.1f}km/h (limit {speed_limit})")
        
        return result
    
    def update(self, detections, class_names, confidences, frame_height, frame_width, stop_threshold=5, email_config=None, distance_meters=None, speed_limit=60):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = [(int((x1+x2)/2), int((y1+y2)/2)) for x1, y1, x2, y2 in detections]
        
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, class_names[i], confidences[i], frame_height, frame_width)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            try:
                from scipy.spatial.distance import cdist
                D = cdist(np.array(object_centroids), np.array(input_centroids), 'euclidean')
            except:
                D = np.zeros((len(object_centroids), len(input_centroids)))
                for i, oc in enumerate(object_centroids):
                    for j, ic in enumerate(input_centroids):
                        D[i][j] = np.sqrt((oc[0]-ic[0])**2 + (oc[1]-ic[1])**2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols or D[row, col] > 80:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                self.check_line_crossing(object_id, input_centroids[col], frame_height, frame_width, distance_meters, speed_limit, email_config)
                
                self.object_history[object_id].append(input_centroids[col])
                if len(self.object_history[object_id]) > 30:
                    self.object_history[object_id].pop(0)
                speed = self.estimate_speed(object_id)
                
                if len(self.object_history[object_id]) >= 10:
                    recent = self.object_history[object_id][-10:]
                    movement = np.std([p[0] for p in recent]) + np.std([p[1] for p in recent])
                    
                    if movement < 5:
                        if not self.object_states[object_id]['is_stopped']:
                            self.object_states[object_id]['is_stopped'] = True
                            self.object_states[object_id]['stop_start_time'] = time.time()
                        else:
                            stop_duration = time.time() - self.object_states[object_id]['stop_start_time']
                            
                            if stop_duration > stop_threshold and object_id not in st.session_state.crossed_objects:
                                st.session_state.crossed_objects.add(object_id)
                                logger.log_event('vehicle_stopped', self.object_states[object_id]['class'], object_id, f"Vehicle stopped for {int(stop_duration)}s", duration=stop_duration)
                                alert_msg = f"STOPPED: {self.object_states[object_id]['class']} ID:{object_id} for {int(stop_duration)}s!"
                                st.session_state.alerts.append(alert_msg)
                                
                                # LOG TO GOOGLE SHEET
                                log_alert_to_sheet(
                                    alert_type="Vehicle Stopped",
                                    details=f"{self.object_states[object_id]['class']} stopped in traffic",
                                    vehicle_id=object_id,
                                    speed="",
                                    duration=f"{int(stop_duration)}s"
                                )
                    else:
                        if self.object_states[object_id]['is_stopped']:
                            self.object_states[object_id]['is_stopped'] = False
                            if object_id in st.session_state.crossed_objects:
                                st.session_state.crossed_objects.remove(object_id)
                
                used_rows.add(row)
                used_cols.add(col)
            
            for col in set(range(len(input_centroids))) - used_cols:
                self.register(input_centroids[col], class_names[col], confidences[col], frame_height, frame_width)
            
            for row in set(range(len(object_ids))) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return self.objects

def get_congestion_level(vehicle_count, max_vehicles=50):
    ratio = vehicle_count / max_vehicles if max_vehicles > 0 else 0
    if ratio <= 0.2:
        return 1, "FREE FLOW", (76, 175, 80)
    elif ratio <= 0.4:
        return 2, "LIGHT", (139, 195, 74)
    elif ratio <= 0.6:
        return 3, "MODERATE", (255, 193, 7)
    elif ratio <= 0.8:
        return 4, "HEAVY", (255, 152, 0)
    else:
        return 5, "CONGESTED", (244, 67, 54)

def generate_pdf_report(events_df, summary_text):
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#8B0000"), spaceAfter=30, alignment=TA_CENTER)
        
        story.append(Paragraph("UTM TRAFFIC MONITORING SYSTEM", title_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Report: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("AI Analysis", styles['Heading2']))
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Traffic Statistics", styles['Heading2']))
        data = [['Metric', 'Value'], ['Events', str(len(events_df) if events_df is not None else 0)], ['Entered', str(st.session_state.cars_entered)], ['Exited', str(st.session_state.cars_exited)], ['Violations', str(len(st.session_state.speed_violations))]]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#8B0000")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None

def generate_ai_summary(time_period_minutes=10):
    if not st.session_state.events_log or gemini_client is None:
        return "No events to summarize."
    cutoff = datetime.now() - timedelta(minutes=time_period_minutes)
    recent = [e for e in st.session_state.events_log if datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S') > cutoff]
    if not recent:
        return f"No events in last {time_period_minutes} minutes."
    events_text = "\n".join([f"- {e['description']}" for e in recent[-15:]])
    prompt = f"""Analyze these traffic events: {events_text}. Provide traffic flow analysis, congestion patterns, speed violations, and recommendations."""
    try:
        response = gemini_client.models.generate_content(model="gemini-2.0-flash-exp", contents=prompt)
        return response.text
    except:
        return "Summary unavailable"

# Sidebar Configuration
st.sidebar.header("Camera Settings")
video_resolution = st.sidebar.selectbox("Video Resolution", ["640x480 (Fast)", "1280x720 (HD)", "1920x1080 (Full HD)"], index=0)
frame_skip = st.sidebar.slider("Process Every N Frames", 1, 10, 5)

if "640x480" in video_resolution:
    video_width, video_height = 640, 480
elif "1280x720" in video_resolution:
    video_width, video_height = 1280, 720
else:
    video_width, video_height = 1920, 1080

st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IoU", 0.1, 1.0, 0.5, 0.05)
stop_threshold = st.sidebar.number_input("Alert if stopped (seconds)", 3, 60, 5)
line_position = st.sidebar.slider("Line 1 Position", 0.2, 0.8, 0.4, 0.05)
line_orientation = st.sidebar.radio("Line Orientation", ["Horizontal", "Vertical"])

st.sidebar.header("Dual Line Speed Detection")
enable_dual_lines = st.sidebar.checkbox("Enable Dual Lines", True)
if enable_dual_lines:
    line2_position = st.sidebar.slider("Line 2 Position", 0.2, 0.8, 0.6, 0.05)
    distance_between_lines = st.sidebar.number_input("Distance between lines (meters)", 1, 500, 10)
else:
    line2_position = None
    distance_between_lines = None

pixels_per_meter = st.sidebar.number_input("Pixels per meter", 1, 50, 10)

st.sidebar.header("Speed Settings")
speed_limit = st.sidebar.number_input("Speed Limit (km/h)", 10, 150, 60)

st.sidebar.header("Parking Settings")
max_parking_vehicles = st.sidebar.number_input("Max Capacity", 5, 500, 50)

st.sidebar.header("Email Alerts (SendGrid)")
email_alerts_enabled = st.sidebar.checkbox("Enable Email Alerts", False)
sender_email = ""
sendgrid_api_key = ""
recipient_email = ""

st.sidebar.header("Google Sheets Logging")
sheets_enabled = st.sidebar.checkbox("Enable Google Sheets", False)
if sheets_enabled:
    st.sidebar.info("📋 Sheet ID: 1R2Fv8_SwIcz4f90qV3i3qaV6FRaN8jRoyAjL11xst1E")
    st.sidebar.warning("⚠️ Requires credentials.json file")
    
    if st.sidebar.button("Test Sheet Connection"):
        test_result = log_alert_to_sheet(
            alert_type="Test",
            details="Connection test successful",
            vehicle_id="TEST",
            speed="",
            duration=""
        )
        if test_result:
            st.sidebar.success("✓ Connected to Google Sheets!")
        else:
            st.sidebar.error("✗ Connection failed. Check credentials.json")

st.sidebar.markdown("---")

if email_alerts_enabled:
    sender_email = st.sidebar.text_input("Your Email Address", type="default", placeholder="your.email@sendgrid.com")
    sendgrid_api_key = st.sidebar.text_input("SendGrid API Key", type="password", placeholder="SG.xxxxxxxxxxxxxxxxxxxxxxxx")
    recipient_email = st.sidebar.text_input("Recipient Email", placeholder="recipient@example.com")
    
    st.sidebar.info("Get API key from SendGrid → Settings → API Keys")
    
    if st.sidebar.button("Test Email", type="primary"):
        if sender_email and sendgrid_api_key and recipient_email:
            test_subject = "UTM Traffic System - Test Email"
            test_message = f"""<html><body style="font-family: Arial, sans-serif;">
                    <h2 style="color: #8B0000;">Test Email Successful!</h2>
                    <p>Your SendGrid email alerts are working correctly!</p>
                    <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </body></html>"""
            
            with st.spinner("Sending test email..."):
                success = send_email_alert(test_subject, test_message, recipient_email, sender_email, sendgrid_api_key)
            
            if success:
                st.sidebar.success("Email sent! Check your inbox.")
            else:
                st.sidebar.error("Failed. Check logs below.")
        else:
            st.sidebar.error("Fill in all fields first!")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Email Status")
    if st.session_state.email_logs:
        for log in st.session_state.email_logs[-5:]:
            if "sent" in log or "✓" in log:
                st.sidebar.success(log)
            else:
                st.sidebar.error(log)
    else:
        st.sidebar.info("No emails sent yet")
    
    if st.sidebar.button("Clear Logs"):
        st.session_state.email_logs = []
        st.rerun()

email_config = {
    'enabled': email_alerts_enabled,
    'sender': sender_email,
    'api_key': sendgrid_api_key,
    'recipient': recipient_email
}

st.sidebar.header("Mode")
mode = st.sidebar.radio("Select", ["Traffic Cam", "Parking Cam"])
st.sidebar.markdown("---")
st.sidebar.info("Created by ABABAKER NAZAR\nv5.2 OPTIMIZED - Dual Line Speed Detection")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Detection", "Analytics", "Events", "Heatmap", "Reports"])

with tab1:
    if st.session_state.alerts:
        for alert in st.session_state.alerts[-3:]:
            if "SPEED" in alert:
                st.markdown(f'<div class="warning-box">{alert}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)
    
    if mode == "Traffic Cam":
        traffic_mode = st.radio("Select Mode", ["Live Camera", "Upload Video"], horizontal=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Entered", st.session_state.cars_entered)
        col2.metric("Exited", st.session_state.cars_exited)
        col3.metric("Violations", len(st.session_state.speed_violations))
        col4.metric("Events", len(st.session_state.events_log))
        col5.metric("Distance", f"{distance_between_lines}m" if distance_between_lines else "N/A")
        
        if traffic_mode == "Live Camera":
            st.subheader("Live Traffic Camera - Dual Line Speed Detection")
            st.info("Two detection lines: Green (Line 1) and Blue (Line 2). Speed calculated between crossings.")
            
            try:
                from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
                
                class TrafficVideoTransformer(VideoTransformerBase):
                    def __init__(self):
                        self.tracker = AdvancedTracker(
                            line_position=line_position, 
                            line_orientation="horizontal" if line_orientation == "Horizontal" else "vertical",
                            line2_position=line2_position,
                            pixels_per_meter=pixels_per_meter
                        )
                        self.frame_count = 0
                        self.last_detection_time = time.time()
                    
                    def transform(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        
                        if img.shape[1] != video_width or img.shape[0] != video_height:
                            img = cv2.resize(img, (video_width, video_height))
                        
                        h, w = img.shape[:2]
                        
                        if line_orientation == "Horizontal":
                            line_y = int(h * line_position)
                            cv2.line(img, (0, line_y), (w, line_y), (0, 255, 0), 2)
                            cv2.putText(img, "Line 1", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            if line2_position:
                                line2_y = int(h * line2_position)
                                cv2.line(img, (0, line2_y), (w, line2_y), (255, 0, 0), 2)
                                cv2.putText(img, "Line 2", (10, line2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        else:
                            line_x = int(w * line_position)
                            cv2.line(img, (line_x, 0), (line_x, h), (0, 255, 0), 2)
                            cv2.putText(img, "Line 1", (line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            if line2_position:
                                line2_x = int(w * line2_position)
                                cv2.line(img, (line2_x, 0), (line2_x, h), (255, 0, 0), 2)
                                cv2.putText(img, "Line 2", (line2_x + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        self.frame_count += 1
                        current_vehicles = 0
                        current_time = time.time()
                        
                        if model and (self.frame_count % 5 == 0 or current_time - self.last_detection_time > 0.5):
                            self.last_detection_time = current_time
                            try:
                                scale = 640 / max(h, w)
                                if scale < 1:
                                    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
                                else:
                                    img_resized = img
                                
                                results = model.predict(img_resized, conf=confidence_threshold, iou=iou_threshold, verbose=False, device='cpu')
                                
                                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                                    boxes = results[0].boxes.xyxy.cpu().numpy()
                                    
                                    if scale < 1:
                                        boxes = boxes / scale
                                    
                                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                                    confidences = results[0].boxes.conf.cpu().numpy()
                                    names = results[0].names
                                    class_names = [names[c] for c in classes]
                                    
                                    current_vehicles = len(boxes)
                                    tracked = self.tracker.update(boxes, class_names, confidences, h, w, stop_threshold, email_config, distance_between_lines, speed_limit)
                                    
                                    for box, cls, conf in zip(boxes, classes, confidences):
                                        x1, y1, x2, y2 = map(int, box)
                                        color = (0, 255, 0)
                                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                        
                                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                        matched_id = None
                                        min_dist = float('inf')
                                        for obj_id, centroid in tracked.items():
                                            dist = (centroid[0]-cx)**2 + (centroid[1]-cy)**2
                                            if dist < 2500 and dist < min_dist:
                                                matched_id = obj_id
                                                min_dist = dist
                                        
                                        speed = 0
                                        if matched_id and matched_id in self.tracker.object_states:
                                            speed = self.tracker.object_states[matched_id]['avg_speed']
                                        
                                        label = f"ID:{matched_id} {speed:.1f}km/h"
                                        cv2.putText(img, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                        
                                        if matched_id and matched_id in self.tracker.object_history:
                                            pts = self.tracker.object_history[matched_id][-5:]
                                            for i in range(1, len(pts)):
                                                cv2.line(img, pts[i-1], pts[i], (255, 0, 0), 2)
                            except Exception as e:
                                pass
                        
                        congestion_level, congestion_text, color_bgr = get_congestion_level(current_vehicles, 50)
                        cv2.putText(img, f"Entered: {st.session_state.cars_entered}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(img, f"Exited: {st.session_state.cars_exited}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(img, congestion_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
                        
                        return img
                
                webrtc_streamer(
                    key="traffic", 
                    video_transformer_factory=TrafficVideoTransformer, 
                    rtc_configuration=RTCConfiguration({
                        "iceServers": [
                            {"urls": ["stun:stun.l.google.com:19302"]},
                            {"urls": ["stun:stun1.l.google.com:19302"]}
                        ]
                    }), 
                    media_stream_constraints={
                        "video": {"width": {"exact": video_width}, "height": {"exact": video_height}, "frameRate": {"ideal": 15}}, 
                        "audio": False
                    }, 
                    async_processing=True,
                    sendback_audio=False
                )
            except ImportError:
                st.error("Install: pip install streamlit-webrtc")
        
        else:
            st.info("Upload a traffic video for analysis with dual-line speed detection")
            uploaded_traffic = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
            
            if uploaded_traffic and model and st.button("Process Video", type="primary"):
                st.session_state.cars_entered = 0
                st.session_state.cars_exited = 0
                st.session_state.speed_violations = []
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_traffic.read())
                    temp_path = tmp.name
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                cap = cv2.VideoCapture(temp_path)
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                fps = int(fps)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if width == 0 or height == 0:
                    st.error("Invalid video dimensions")
                    cap.release()
                    st.stop()
                
                output_path = "output_annotated.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    st.error("Video codec unavailable")
                    cap.release()
                    st.stop()
                
                heatmap = np.zeros((height, width), dtype=np.float32)
                tracker = AdvancedTracker(
                    line_position=line_position,
                    line_orientation="horizontal" if line_orientation == "Horizontal" else "vertical",
                    line2_position=line2_position,
                    pixels_per_meter=pixels_per_meter
                )
                
                frame_count = 0
                
                if line_orientation == "Horizontal":
                    line_y = int(height * line_position)
                    line2_y = int(height * line2_position) if line2_position else None
                else:
                    line_x = int(width * line_position)
                    line2_x = int(width * line2_position) if line2_position else None
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    if line_orientation == "Horizontal":
                        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 1)
                        if line2_y:
                            cv2.line(frame, (0, line2_y), (width, line2_y), (255, 0, 0), 1)
                    else:
                        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 0), 1)
                        if line2_x:
                            cv2.line(frame, (line2_x, 0), (line2_x, height), (255, 0, 0), 1)
                    
                    if frame_count % 2 == 0:
                        try:
                            results = model.predict(frame, conf=confidence_threshold, iou=iou_threshold, verbose=False)
                            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                                boxes = results[0].boxes.xyxy.cpu().numpy()
                                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                                confidences = results[0].boxes.conf.cpu().numpy()
                                names = results[0].names
                                class_names = [names[c] for c in classes]
                                
                                current_vehicles = len(boxes)
                                congestion_level, congestion_text, color_bgr = get_congestion_level(current_vehicles, 50)
                                
                                tracked = tracker.update(boxes, class_names, confidences, height, width, stop_threshold, email_config, distance_between_lines, speed_limit)
                                
                                for box, cls, conf in zip(boxes, classes, confidences):
                                    x1, y1, x2, y2 = map(int, box)
                                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                                    
                                    if x1 < x2 and y1 < y2:
                                        heatmap[y1:y2, x1:x2] += 1
                                        color = (0, 255, 0)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                        
                                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                        matched_id = None
                                        min_dist = float('inf')
                                        for obj_id, centroid in tracked.items():
                                            dist = (centroid[0]-cx)**2 + (centroid[1]-cy)**2
                                            if dist < 6400 and dist < min_dist:
                                                matched_id = obj_id
                                                min_dist = dist
                                        
                                        speed = 0
                                        if matched_id and matched_id in tracker.object_states:
                                            speed = tracker.object_states[matched_id]['avg_speed']
                                        
                                        label = f"ID:{matched_id} {speed:.1f}km/h"
                                        cv2.putText(frame, label, (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        
                                        if matched_id and matched_id in tracker.object_history:
                                            pts = tracker.object_history[matched_id][-10:]
                                            for i in range(1, len(pts)):
                                                cv2.line(frame, pts[i-1], pts[i], (255, 0, 0), 2)
                                
                                cv2.putText(frame, congestion_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_bgr, 3)
                        except Exception as e:
                            pass
                    
                    cv2.putText(frame, f"Entered: {st.session_state.cars_entered}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f"Exited: {st.session_state.cars_exited}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    out.write(frame)
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    status_text.text(f"Frame {frame_count}/{total_frames}")
                
                cap.release()
                out.release()
                logger.flush()
                st.session_state.heatmap_data = heatmap
                
                st.success(f"Processed {frame_count} frames!")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Entered", st.session_state.cars_entered)
                col2.metric("Exited", st.session_state.cars_exited)
                col3.metric("Violations", len(st.session_state.speed_violations))
                col4.metric("Tracked", tracker.next_object_id)
                
                st.subheader("Annotated Video")
                st.video(output_path)
                with open(output_path, 'rb') as f:
                    st.download_button("Download Video", f, "annotated.mp4", "video/mp4")
    
    elif mode == "Parking Cam":
        st.subheader("Parking Lot Management")
        
        parking_mode = st.radio("Select Mode", ["Live Camera", "Upload Video"], horizontal=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Capacity", max_parking_vehicles)
        col2.metric("Parked", len(st.session_state.parking_vehicles))
        col3.metric("Available", max_parking_vehicles - len(st.session_state.parking_vehicles))
        
        pct = (len(st.session_state.parking_vehicles) / max_parking_vehicles) * 100 if max_parking_vehicles > 0 else 0
        available_pct = 100 - pct
        
        if len(st.session_state.parking_vehicles) >= max_parking_vehicles:
            st.markdown('<div class="alert-box">Parking FULL!</div>', unsafe_allow_html=True)
            # LOG TO GOOGLE SHEET
            log_alert_to_sheet(
                alert_type="Parking Full",
                details=f"Parking lot reached maximum capacity ({max_parking_vehicles} vehicles)",
                vehicle_id="",
                speed="",
                duration=""
            )
        else:
            if pct > 80:
                st.markdown(f'<div class="warning-box">{pct:.0f}% FULL - Limited Space</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box">{available_pct:.0f}% AVAILABLE</div>', unsafe_allow_html=True)
        
        if parking_mode == "Live Camera":
            st.info("Real-time parking lot monitoring with vehicle tracking")
            
            try:
                from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
                
                class ParkingVideoTransformer(VideoTransformerBase):
                    def __init__(self):
                        self.tracker = AdvancedTracker()
                        self.frame_count = 0
                        self.parking_alert_sent = False
                        self.last_detection_time = time.time()
                    
                    def transform(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        
                        if img.shape[1] != video_width or img.shape[0] != video_height:
                            img = cv2.resize(img, (video_width, video_height))
                        
                        h, w = img.shape[:2]
                        self.frame_count += 1
                        current_vehicles = set()
                        current_time = time.time()
                        
                        if model and (self.frame_count % frame_skip == 0 or current_time - self.last_detection_time > 1.0):
                            self.last_detection_time = current_time
                            try:
                                detect_scale = 416 / max(h, w)
                                if detect_scale < 1:
                                    img_detect = cv2.resize(img, (int(w * detect_scale), int(h * detect_scale)))
                                else:
                                    img_detect = img
                                
                                results = model.predict(img_detect, conf=confidence_threshold, iou=iou_threshold, verbose=False, device='cpu')
                                
                                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                                    boxes = results[0].boxes.xyxy.cpu().numpy()
                                    
                                    if detect_scale < 1:
                                        boxes = boxes / detect_scale
                                    
                                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                                    confidences = results[0].boxes.conf.cpu().numpy()
                                    names = results[0].names
                                    class_names = [names[c] for c in classes]
                                    
                                    tracked = self.tracker.update(boxes, class_names, confidences, h, w)
                                    
                                    for obj_id in tracked.keys():
                                        current_vehicles.add(obj_id)
                                    
                                    st.session_state.parking_vehicles = current_vehicles
                                    
                                    for box, cls, conf in zip(boxes, classes, confidences):
                                        x1, y1, x2, y2 = map(int, box)
                                        
                                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                        matched_id = None
                                        min_dist = float('inf')
                                        for obj_id, centroid in tracked.items():
                                            dist = (centroid[0]-cx)**2 + (centroid[1]-cy)**2
                                            if dist < 2500 and dist < min_dist:
                                                matched_id = obj_id
                                                min_dist = dist
                                        
                                        color = (0, 255, 0)
                                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                        
                                        label = f"ID:{matched_id} {names[cls]}"
                                        cv2.putText(img, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    
                                    if len(current_vehicles) >= max_parking_vehicles and not self.parking_alert_sent:
                                        self.parking_alert_sent = True
                                        st.session_state.alerts.append(f"PARKING FULL: {len(current_vehicles)}/{max_parking_vehicles}")
                                    elif len(current_vehicles) < max_parking_vehicles:
                                        self.parking_alert_sent = False
                            except Exception as e:
                                pass
                        
                        parked = len(st.session_state.parking_vehicles)
                        available = max_parking_vehicles - parked
                        pct = (parked / max_parking_vehicles * 100) if max_parking_vehicles > 0 else 0
                        
                        status_color = (0, 255, 0) if pct < 80 else (0, 165, 255) if pct < 100 else (0, 0, 255)
                        
                        cv2.putText(img, f"Parked: {parked}/{max_parking_vehicles}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                        cv2.putText(img, f"Available: {available}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(img, f"Occupancy: {pct:.0f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                        
                        if parked >= max_parking_vehicles:
                            cv2.putText(img, "FULL!", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        
                        return img
                
                webrtc_streamer(
                    key="parking_live", 
                    video_transformer_factory=ParkingVideoTransformer, 
                    rtc_configuration=RTCConfiguration({
                        "iceServers": [
                            {"urls": ["stun:stun.l.google.com:19302"]},
                            {"urls": ["stun:stun1.l.google.com:19302"]}
                        ]
                    }), 
                    media_stream_constraints={
                        "video": {"width": {"exact": video_width}, "height": {"exact": video_height}, "frameRate": {"ideal": 15}},
                        "audio": False
                    }, 
                    async_processing=True,
                    sendback_audio=False
                )
            except ImportError:
                st.error("Install: pip install streamlit-webrtc")
        
        else:
            st.info("Upload a parking video for complete analysis")
            uploaded_parking = st.file_uploader("Upload parking video", type=["mp4", "avi", "mov"])
            
            if uploaded_parking and model and st.button("Analyze Video", type="primary"):
                st.session_state.parking_vehicles = set()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_parking.read())
                    temp_path = tmp.name
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                cap = cv2.VideoCapture(temp_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                tracker = AdvancedTracker()
                count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    count += 1
                    
                    if count % 3 != 0:
                        continue
                    
                    try:
                        results = model.predict(frame, conf=confidence_threshold, iou=iou_threshold, verbose=False)
                        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                            classes = results[0].boxes.cls.cpu().numpy().astype(int)
                            confidences = results[0].boxes.conf.cpu().numpy()
                            names = results[0].names
                            class_names = [names[c] for c in classes]
                            h, w = frame.shape[:2]
                            tracked = tracker.update(boxes, class_names, confidences, h, w)
                            for obj_id in tracked.keys():
                                st.session_state.parking_vehicles.add(obj_id)
                    except:
                        pass
                    
                    progress_bar.progress(min(count / total, 1.0))
                    status_text.text(f"Processing frame {count}/{total}")
                
                cap.release()
                logger.flush()
                
                detected_vehicles = len(st.session_state.parking_vehicles)
                st.success(f"Analysis Complete: Detected {detected_vehicles} vehicles")
                
                if detected_vehicles >= max_parking_vehicles:
                    st.markdown('<div class="alert-box">Parking FULL!</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Analytics Dashboard")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Entered", st.session_state.cars_entered)
    col2.metric("Exited", st.session_state.cars_exited)
    col3.metric("Events", len(st.session_state.events_log))
    col4.metric("Violations", len(st.session_state.speed_violations))
    col5.metric("Parked", len(st.session_state.parking_vehicles))
    
    # Congestion Level Alert
    if st.session_state.events_log:
        df = pd.DataFrame(st.session_state.events_log)
        
        # Calculate current congestion
        current_vehicles = st.session_state.cars_entered - st.session_state.cars_exited
        congestion_level, congestion_text, color_bgr = get_congestion_level(current_vehicles, 50)
        
        # Display congestion alert
        if congestion_level >= 4:
            st.markdown(f'<div class="alert-box">🚨 CONGESTION LEVEL: {congestion_text} ({current_vehicles} vehicles)</div>', unsafe_allow_html=True)
        elif congestion_level == 3:
            st.markdown(f'<div class="warning-box">⚠️ CONGESTION LEVEL: {congestion_text} ({current_vehicles} vehicles)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ CONGESTION LEVEL: {congestion_text} ({current_vehicles} vehicles)</div>', unsafe_allow_html=True)
        
        st.subheader("Event Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.write("By Class")
            st.bar_chart(df['object_class'].value_counts())
        with col2:
            st.write("By Type")
            st.bar_chart(df['event_type'].value_counts())
        
        st.subheader("Recent Events")
        st.dataframe(df.tail(15), use_container_width=True)
    else:
        st.info("No analytics data available yet")

with tab3:
    st.subheader("Event Timeline")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.events_log:
            # Filter to show only relevant events
            relevant_events = ['vehicle_detected', 'vehicle_entered', 'vehicle_exited', 'vehicle_stopped', 'speed_violation', 'parking_full']
            filtered_events = [e for e in st.session_state.events_log if e['event_type'] in relevant_events]
            
            for event in reversed(filtered_events[-20:]):
                if event['event_type'] == 'vehicle_entered':
                    icon = "🚗➡️"
                elif event['event_type'] == 'vehicle_exited':
                    icon = "🚗⬅️"
                elif event['event_type'] == 'vehicle_stopped':
                    icon = "⏹️"
                elif event['event_type'] == 'speed_violation':
                    icon = "⚠️"
                elif event['event_type'] == 'parking_full':
                    icon = "🅿️"
                else:
                    icon = "📍"
                
                st.markdown(f"{icon} **{event['timestamp']}** | {event['object_class']} ID:{event['object_id']}")
                st.write(f"{event['description']}")
                if event['speed'] > 0:
                    st.write(f"Speed: {event['speed']:.1f} km/h")
                st.markdown("---")
        else:
            st.info("No events yet")
    
    with col2:
        st.subheader("AI Summary")
        time_range = st.selectbox("Period (min)", [5, 10, 30])
        if st.button("Generate", use_container_width=True):
            with st.spinner("Analyzing..."):
                summary = generate_ai_summary(time_range)
                st.success(summary)
        
        if st.button("Export CSV", use_container_width=True):
            if st.session_state.events_log:
                logger.flush()
                df = pd.DataFrame(st.session_state.events_log)
                st.download_button("Download", df.to_csv(index=False), "events.csv", "text/csv")

with tab4:
    st.subheader("Activity Heatmap")
    if st.session_state.heatmap_data is not None:
        heatmap = st.session_state.heatmap_data
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        st.image(heatmap_colored, caption="Activity Zones", channels="BGR", use_container_width=True)
        st.info("Red = High Activity | Blue = Low Activity")
    else:
        st.info("Process a video to generate heatmap")

with tab5:
    st.subheader("PDF Report")
    if st.session_state.events_log or st.session_state.cars_entered > 0:
        report_time = st.selectbox("Period (min)", [5, 10, 30, 60])
        if st.button("Generate PDF", type="primary", use_container_width=True):
            with st.spinner("Generating..."):
                logger.flush()
                summary = generate_ai_summary(report_time)
                events_df = pd.DataFrame(st.session_state.events_log) if st.session_state.events_log else None
                pdf_data = generate_pdf_report(events_df, summary)
                if pdf_data:
                    st.success("PDF Ready!")
                    st.download_button("Download PDF", pdf_data, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf")
    else:
        st.info("No data for report yet")

st.markdown("""
<div class="footer">
    <p style="font-size: 1.4rem; margin-bottom: 10px;">UTM TRAFFIC MONITORING SYSTEM v5.2</p>
    <p style="font-size: 1.1rem; margin-bottom: 5px;">Created by ABABAKER NAZAR</p>
    <p style="font-size: 0.95rem; opacity: 0.8;">Dual-Line Speed Detection • Vehicle Tracking • Real-time Analysis</p>
    <p style="color: #8B0000; font-weight: bold; font-size: 1.2rem; margin-top: 15px;">OPTIMIZED PERFORMANCE</p>
</div>

""", unsafe_allow_html=True)
