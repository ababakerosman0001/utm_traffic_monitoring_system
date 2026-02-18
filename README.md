# UTM TRAFFIC MONITORING SYSTEM

<div align="center">

```
 _   _ _____ __  __   _____ ____      _    _____ _____ ___ ____
| | | |_   _|  \/  | |_   _|  _ \    / \  |  ___|  ___|_ _/ ___|
| | | | | | | |\/| |   | | | |_) |  / _ \ | |_  | |_   | | |
| |_| | | | | |  | |   | | |  _ <  / ___ \|  _| |  _|  | | |___
 \___/  |_| |_|  |_|   |_| |_| \_\/_/   \_\_|   |_|   |___\____|

      M O N I T O R I N G   S Y S T E M   v 5 . 2   O P T I M I Z E D
```

**Advanced Vehicle Detection  |  Dual-Line Speed Estimation  |  AI-Powered Analysis**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com)
[![Gemini AI](https://img.shields.io/badge/Gemini_2.0-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-8B0000?style=for-the-badge)](LICENSE)

*Created by **ABABAKER NAZAR** | v5.2 OPTIMIZED -- 3-5x Faster*

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [Dual-Line Speed Detection](#dual-line-speed-detection)
- [AI-Powered Analysis](#ai-powered-analysis)
- [Alert System](#alert-system)
- [Operating Modes](#operating-modes)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [UI Design System](#ui-design-system)
- [Project Structure](#project-structure)

---

## Overview

The **UTM Traffic Monitoring System** is a production-grade, real-time traffic intelligence platform built for Universiti Teknologi Malaysia. It combines state-of-the-art computer vision with agentic AI to deliver actionable insights from any camera feed or uploaded video.

The system operates across two primary modes -- **Traffic Camera** and **Parking Camera** -- with a unified dashboard, multi-channel alert engine, and automated PDF reporting powered by Google Gemini 2.0.

---

## System Architecture

```
+------------------------------------------------------------------+
|                        SYSTEM PIPELINE                           |
|                                                                  |
|  [Video Input]  -->  [YOLOv8]      -->  [AdvancedTracker]       |
|  (Live/Upload)       Detection          Centroid + IoU           |
|                           |                     |                |
|                           v                     v                |
|                      [Speed              [Alert Engine]         |
|                       Estimation]         Email + Sheets         |
|                           |                     |                |
|                           v                     v                |
|                      [Gemini AI]          [Dashboard +          |
|                       Summaries]           PDF Reports]          |
+------------------------------------------------------------------+
```

The app runs within a **Streamlit web interface** with 5 intelligent tabs:

| Tab | Purpose | Key Capability |
|-----|---------|----------------|
| Detection | Live/Video processing | Real-time tracking and speed display |
| Analytics | Charts and metrics | Event distribution, congestion level |
| Events | Timeline log | Filterable event history with AI summary |
| Heatmap | Activity zones | Spatial activity visualization |
| Reports | PDF export | AI-written traffic analysis reports |

---

## Core Features

### Vehicle Detection and Tracking

```
Detection Engine:  YOLOv8 (custom trained model: best.pt)
Tracker:           Custom AdvancedTracker (centroid + IoU matching)
Max Lost Frames:   30 frames before a vehicle is deregistered
History Buffer:    30-point centroid trail per tracked vehicle
Speed Buffer:      10-sample rolling average per vehicle
```

- Real-time bounding boxes with persistent vehicle ID labels
- Visual trail rendering showing the last 5-10 movement points per vehicle
- Centroid-based identity persistence across consecutive frames
- Frame-skip optimization: process every Nth frame (configurable 1-10)
- Auto-scales inference resolution for performance (416px or 640px)

---

## Dual-Line Speed Detection

The system's flagship feature -- **physics-based speed measurement** using two configurable virtual detection lines placed across the camera view.

```
+--------------------------------------------------+
|                                                  |
|   Vehicle >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        |
|                                                  |
| ================ LINE 1 (Green) ==============   |  <-- t1 recorded
|                                                  |
|   [  real-world distance configured by user ]    |
|                                                  |
| ================ LINE 2 (Blue)  ==============   |  <-- t2 recorded
|                                                  |
|   Speed = (distance_meters / delta_t) x 3.6      |
+--------------------------------------------------+
```

**How it works:**

1. Vehicle crosses **Line 1** -- timestamp `t1` is recorded
2. Vehicle crosses **Line 2** -- timestamp `t2` is recorded
3. Speed is calculated: `speed_kmh = (meters / (t2 - t1)) x 3.6`
4. If speed exceeds the configured limit, an alert fires instantly

**Configurable parameters:**

| Setting | Range | Default |
|---------|-------|---------|
| Line orientation | Horizontal / Vertical | Horizontal |
| Line 1 position | 20% to 80% of frame | 40% |
| Line 2 position | 20% to 80% of frame | 60% |
| Distance between lines | 1-500 meters | 10 m |
| Speed limit | 10-150 km/h | 60 km/h |

---

## AI-Powered Analysis

Powered by **Google Gemini 2.0 Flash** for intelligent, narrative-form traffic reporting.

```python
# Events are collected over a configurable time window,
# then sent to Gemini 2.0 Flash for analysis

prompt = """
Analyze these traffic events:
  - Vehicle detected, ID:12
  - Speed violation: 87 km/h (limit 60)
  - Vehicle stopped for 23s

Provide: traffic flow analysis, congestion patterns,
         speed violations, and recommendations.
"""
```

**Capabilities:**

- Natural language traffic summaries over 5, 10, or 30-minute windows
- Congestion pattern recognition from raw event streams
- Speed violation analysis with contextual commentary
- Actionable recommendations for traffic management decisions
- AI-written summaries embedded directly into downloadable PDF reports

---

## Alert System

Three integrated notification pathways trigger on critical events.

### Email Alerts via SendGrid

```
Trigger:   Vehicle exceeds the configured speed limit

Content:
  - Vehicle ID
  - Measured speed vs. configured limit
  - Timestamp
  - HTML formatted alert body

API:       SendGrid REST API v3
```

### Google Sheets Logging

```
Sheet ID: 1R2Fv8_SwIcz4f90qV3i3qaV6FRaN8jRoyAjL11xst1E

Logged event types:
  +------------------+-------------------------------------+
  | Alert Type       | Example Detail                      |
  +------------------+-------------------------------------+
  | Speed Violation  | Exceeded limit by X km/h            |
  | Vehicle Stopped  | Stopped in traffic for X seconds    |
  | Parking Full     | Lot reached maximum capacity        |
  +------------------+-------------------------------------+

Columns: Timestamp | Type | Vehicle ID | Details | Speed | Duration
```

### In-App Alert Banners

```
Red banner    --> Vehicle stopped / Parking full  (critical)
Yellow banner --> Speed violation                 (warning)
Green banner  --> Normal traffic flow             (info)
```

---

## Operating Modes

### Mode 1: Traffic Camera

#### Live Camera (WebRTC)

- Browser-based real-time stream via `streamlit-webrtc`
- WebRTC with Google STUN servers for NAT traversal
- Configurable resolution: `640x480` | `1280x720` | `1920x1080`
- 15 FPS target with asynchronous frame processing
- Per-vehicle speed displayed in real time on the video feed

#### Video Upload

- Accepts: `.mp4`, `.avi`, `.mov`, `.mkv`
- Full annotation: bounding boxes, persistent IDs, speed values
- Progress bar with live frame counter during processing
- Heatmap generation from detection accumulation across all frames
- One-click download of the fully annotated output video

---

### Mode 2: Parking Camera

Real-time parking lot occupancy management with automatic capacity alerts.

```
Capacity tracking:  Configurable (5 to 500 spaces)
Occupancy display:  Live vehicle count + percentage fill
Status thresholds:  Green (<80%) | Orange (80-99%) | Red (FULL)
```

| Occupancy Level | Status | System Action |
|-----------------|--------|---------------|
| Below 80% | Available | Normal green display |
| 80% to 99% | Limited Space | Warning banner shown |
| 100% | FULL | Alert banner + Email + Sheet log |

---

## Tech Stack

```
+-------------------+--------------------------------------------+
| Layer             | Technology                                 |
+-------------------+--------------------------------------------+
| UI Framework      | Streamlit 1.x                              |
| Computer Vision   | OpenCV 4.x                                 |
| Object Detection  | Ultralytics YOLOv8 (custom best.pt)        |
| AI / LLM          | Google Gemini 2.0 Flash                    |
| Object Tracking   | Custom AdvancedTracker (SciPy cdist)       |
| Email             | SendGrid REST API v3                       |
| Sheets            | gspread + Google OAuth2 Service Account    |
| PDF Reports       | ReportLab                                  |
| Video Streaming   | streamlit-webrtc (WebRTC)                  |
| Data Processing   | Pandas, NumPy                              |
| Image Utilities   | Pillow                                     |
+-------------------+--------------------------------------------+
```

---

## Installation

### Prerequisites

- Python 3.8+
- Webcam (for live camera mode)
- Google Gemini API key
- SendGrid API key (optional, for email alerts)
- `credentials.json` from Google Cloud Console (optional, for Sheets)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/ababakerosman0001/utm-traffic-monitoring.git
cd utm-traffic-monitoring

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your YOLO model weights
cp /path/to/your/best.pt ./best.pt

# 4. (Optional) Add Google Sheets service account credentials
cp /path/to/credentials.json ./credentials.json

# 5. Run the application
streamlit run app.py
```

### requirements.txt

```
streamlit
opencv-python
ultralytics
streamlit-webrtc
google-genai
gspread
google-auth
pandas
numpy
pillow
reportlab
scipy
requests
```

---

## Configuration

All settings are controlled from the **sidebar** with live updates -- no application restart required.

### Camera Settings

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Video Resolution | 3 presets | 640x480 | Processing resolution |
| Frame Skip | 1-10 | 5 | Process every Nth frame |

### Detection Settings

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Confidence | 0.1-1.0 | 0.50 | YOLO detection confidence threshold |
| IoU | 0.1-1.0 | 0.50 | Intersection-over-union threshold |
| Stop Alert | 3-60s | 5s | Seconds before a stopped-vehicle alert fires |
| Pixels per Meter | 1-50 | 10 | Camera calibration for pixel-based speed |

### Speed Detection

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Line 1 Position | 20%-80% | 40% | First detection line placement |
| Line 2 Position | 20%-80% | 60% | Second detection line placement |
| Line Distance | 1-500m | 10m | Real-world distance between lines |
| Speed Limit | 10-150 km/h | 60 | Violation trigger threshold |

---

## UI Design System

### Color Palette

```
Primary:     #8B0000  -- Deep crimson (metrics, active tabs, headings)
Secondary:   #A00000  -- Rich red (cards, chart backgrounds)
Background:  #FAF7F2  -- Warm cream (main page background)
Surface:     #FFFBF5  -- Off-white (tab bar, input fields)
Warning:     #E8DCC6  -- Warm tan (warning banners)
```

### Typography

```
Display:  Orbitron 900    -- Futuristic numerics, system headers
Body:     Rajdhani 500/700 -- UI labels, descriptions, alert text
```

### Congestion Level Scale

```
Level 1 -- FREE FLOW    #4CAF50  (Green)    0-20%  vehicle density
Level 2 -- LIGHT        #8BC34A  (Lt Green) 20-40% vehicle density
Level 3 -- MODERATE     #FFC107  (Amber)    40-60% vehicle density
Level 4 -- HEAVY        #FF9800  (Orange)   60-80% vehicle density
Level 5 -- CONGESTED    #F44336  (Red)      80-100% vehicle density
```

---

## Project Structure

```
utm-traffic-monitoring/
|
+-- app.py                  Main Streamlit application
+-- best.pt                 Custom YOLOv8 model weights
+-- credentials.json        Google Sheets OAuth (optional)
+-- events_log.csv          Auto-generated event log (created at runtime)
+-- output_annotated.mp4    Annotated video output (created at runtime)
+-- requirements.txt
+-- README.md
```

### Key Classes and Functions

```python
AdvancedTracker              # Multi-object centroid tracker
  register()                 # Register new vehicle, determine entry side
  deregister()               # Clean up state for lost vehicle
  check_line_crossing()      # Dual-line speed measurement logic
  estimate_speed()           # Pixel-based rolling average speed fallback
  update()                   # Main update loop with IoU matching

EventLogger                  # Buffered CSV event logging
  log_event()                # Append event to buffer and session state
  flush()                    # Write buffer batch to disk

generate_ai_summary()        # Gemini 2.0 Flash traffic narrative
generate_pdf_report()        # ReportLab PDF with AI-written content
send_email_alert()           # SendGrid v3 API email dispatch
log_alert_to_sheet()         # gspread Google Sheets row append
get_congestion_level()       # 5-level density classification
```

---

## Speed Estimation Methods

### Method 1 -- Dual-Line (Primary, Accurate)

```
speed_kmh = (distance_meters / time_delta_seconds) x 3.6
```

Activated when a vehicle crosses both virtual lines in sequence. Accuracy depends on correct calibration of the real-world distance between lines.

### Method 2 -- Pixel Displacement (Secondary, Approximate)

```
pixel_distance = sqrt(dx^2 + dy^2)
real_distance  = pixel_distance / pixels_per_meter
speed_kmh      = real_distance x 3.6  (averaged over last 10 frames)
```

Used as a fallback display value when dual-line crossing data is not yet available for a newly tracked vehicle.

---

## Event Types Logged

| Event | Trigger Condition | Severity |
|-------|-------------------|----------|
| `vehicle_detected` | New object appears in frame | Info |
| `vehicle_entered` | Vehicle crosses Line 1 | Info |
| `vehicle_exited` | Vehicle crosses Line 2 | Info |
| `vehicle_stopped` | No movement for more than N seconds | Warning |
| `speed_violation` | Measured speed exceeds configured limit | Critical |
| `parking_full` | Tracked vehicles reach max capacity | Critical |

---

## Google Sheets Integration

Alerts are automatically appended to a shared Google Sheet for team-wide monitoring:

```
Sheet URL:
  docs.google.com/spreadsheets/d/1R2Fv8_SwIcz4f90qV3i3qaV6FRaN8jRoyAjL11xst1E

Column layout:
  A: Timestamp   (YYYY-MM-DD HH:MM:SS)
  B: Alert Type  (Speed Violation / Vehicle Stopped / Parking Full)
  C: Vehicle ID  (integer tracker ID)
  D: Details     (human-readable description)
  E: Speed       (km/h, where applicable)
  F: Duration    (seconds, where applicable)
```

Setup: Place a `credentials.json` Google Service Account file in the project root, enable **Google Sheets Logging** in the sidebar, and use the built-in **Test Sheet Connection** button to verify the integration before starting a monitoring session.

---

## PDF Report Generation

Each generated report includes:

- Cover header with generation timestamp
- Full AI-generated analysis narrative from Gemini 2.0 Flash
- Traffic statistics summary table (entered / exited / violations / events)
- UTM branding with crimson color scheme
- One-click download from the Reports tab

---

<div align="center">

---

**UTM TRAFFIC MONITORING SYSTEM v5.2**

Dual-Line Speed Detection  |  Vehicle Tracking  |  Real-time AI Analysis

Built by **ABABAKER NAZAR**

`OPTIMIZED PERFORMANCE -- 3-5x FASTER`

---

</div>
