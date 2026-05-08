import cv2
import base64
import numpy as np
import pandas as pd
import random
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objs as go
from collections import deque
from datetime import datetime

# --- CONFIGURATION ---
VALID_USER = "lakshnaasree"
VALID_PASS = "031206"
CYAN_GLOW = "#00e5ff"    
ALERT_RED = "#ff4d4d"
OK_GREEN = "#00e676"
SEA_BLUE_BG = "#001b2e"

app = Dash(__name__)
app.title = "DeepSea AI.Core"

# Data Buffers
history = {k: deque([0]*30, maxlen=30) for k in ['temp', 'press', 'volt', 'curr']}
prev_gray = None  
mission_log_buffer = []

# --- MOTION ANALYSIS (WAVE FORCE) ---
camera = cv2.VideoCapture(0)

def get_motion_metrics():
    global prev_gray
    success, frame = camera.read()
    if not success: return None, 0, "OFFLINE"
    
    frame = cv2.resize(frame, (400, 300))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    force = 0
    if prev_gray is not None:
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        force = (np.sum(thresh) / 255) / 40 
    
    prev_gray = gray
    blue_tint = np.full(frame.shape, (200, 100, 0), dtype='uint8')
    frame = cv2.addWeighted(frame, 0.8, blue_tint, 0.2, 0)
    
    _, buffer = cv2.imencode('.jpg', frame)
    status = "TURBULENT" if force > 12 else "STABLE"
    
    return base64.b64encode(buffer).decode('utf-8'), force, status

# --- UI LAYOUT ---
app.layout = html.Div([
    # 🌊 DECORATIVE BACKGROUND WITH ANIMATIONS
    html.Div([
        # Bubbles
        *[html.Div(className="bubble", style={"left": f"{random.randint(5, 95)}%", "width": f"{random.randint(5, 15)}px", "height": f"{random.randint(5, 15)}px", "animationDelay": f"{random.randint(0, 10)}s"}) for _ in range(15)],
        
        # Swimming Fish
        html.Div("🐟", className="fish f1"),
        html.Div("🐠", className="fish f2"),
        html.Div("🐡", className="fish f3"),

        # Patrolling Submarine
        html.Div([
            html.Div("🚢", className="submarine"), 
            html.Div(className="exhaust")
        ], className="sub-container"),
    ], className="ocean-overlay"),

    # 1. AUTHENTICATION PAGE
    html.Div([
        html.Div([
            html.Div([
                html.H2("DEEPSEA COMMAND AUTHENTICATION", style={"color": CYAN_GLOW, "marginBottom": "5px"}),
                html.P("RESTRICTED ACCESS - LEVEL 5 CLEARANCE", style={"color": "#00b4d8", "fontSize": "12px", "marginTop": "0"}),
            ], style={"borderBottom": f"1px solid {CYAN_GLOW}", "marginBottom": "35px", "paddingBottom": "15px"}),
            
            html.Div([
                html.Label("OPERATOR IDENTITY", className="login-label"),
                dcc.Input(id="username", type="text", placeholder="Enter Operator ID", className="deep-input"),
                
                html.Label("ACCESS PROFESSION", className="login-label"),
                dcc.Dropdown(
                    id="profession",
                    options=[
                        {'label': 'Marine Engineer', 'value': 'Engineer'},
                        {'label': 'Core Navigator', 'value': 'Navigator'},
                        {'label': 'AI Systems Specialist', 'value': 'Specialist'},
                        {'label': 'Mission Commander', 'value': 'Commander'}
                    ],
                    placeholder="Select Profession",
                    className="deep-dropdown"
                ),

                html.Label("ENCRYPTION KEY", className="login-label"),
                dcc.Input(id="password", type="password", placeholder="Enter Access Code", className="deep-input"),
                
                html.Button("INITIALIZE MISSION CORE", id="login-btn", n_clicks=0, className="glow-button")
            ], style={"textAlign": "left"})
        ], className="login-box")
    ], id="login-screen", className="overlay-flex"),

    # 2. MAIN DASHBOARD
    html.Div([
        html.Div([
            html.H1("⚓ DEEPSEA AI.CORE - COMMAND STACK", style={"color": CYAN_GLOW, "margin":"0"}),
            html.Div(id="operator-display", style={"color": "#00b4d8", "fontWeight": "bold"}),
            html.Button("📥 EXPORT MISSION LOG (.CSV)", id="btn-download", className="nav-btn"),
            dcc.Download(id="download-log-csv"),
        ], style={"textAlign": "center", "padding": "15px", "borderBottom":f"1px solid {CYAN_GLOW}", "marginBottom":"20px"}),

        html.Div([
            # LEFT: WAVE SCANNER
            html.Div([
                html.Div([
                    html.H4("BIOMETRIC WAVE SCANNER", style={"color":CYAN_GLOW, "textAlign":"center"}),
                    html.Img(id="webcam-feed", className="webcam-frame"),
                    html.Div(id="force-val", className="force-text"),
                    html.Div(id="status-box", className="status-indicator"),
                ], className="scanner-wrap"),
                html.Div(id="solution-panel", className="alert-panel"),
            ], style={"flex": "1"}),

            # RIGHT: TELEMETRY STACK
            html.Div([
                html.Div([
                    html.Div([html.Label("THERMAL CORE"), html.H2(id="d-temp", className="val-glow")], className="stat-card"),
                    dcc.Graph(id="g-temp", config={'displayModeBar': False}, className="stack-graph")
                ], className="telemetry-row"),
                html.Div([
                    html.Div([html.Label("HULL PRESSURE"), html.H2(id="d-press", className="val-glow")], className="stat-card"),
                    dcc.Graph(id="g-press", config={'displayModeBar': False}, className="stack-graph")
                ], className="telemetry-row"),
                html.Div([
                    html.Div([html.Label("CORE VOLTAGE"), html.H2(id="d-volt", className="val-glow")], className="stat-card"),
                    dcc.Graph(id="g-volt", config={'displayModeBar': False}, className="stack-graph")
                ], className="telemetry-row"),
                html.Div([
                    html.Div([html.Label("POWER CURRENT"), html.H2(id="d-curr", className="val-glow")], className="stat-card"),
                    dcc.Graph(id="g-curr", config={'displayModeBar': False}, className="stack-graph")
                ], className="telemetry-row"),
            ], style={"flex": "2", "display":"flex", "flexDirection":"column", "gap":"15px"}),
        ], style={"display": "flex", "gap": "30px", "padding":"0 20px"}),

        dcc.Interval(id="timer", interval=2500, n_intervals=0),
        dcc.Interval(id="cam-timer", interval=100, n_intervals=0),
        dcc.Store(id='voice-store'),
    ], id="main-content", style={"display": "none"}),

    html.Div(id="voice-trigger", style={"display": "none"})
], style={"backgroundColor": SEA_BLUE_BG, "minHeight": "100vh", "position": "relative", "overflow": "hidden"})

# --- CALLBACKS ---

@app.callback(
    [Output("login-screen", "style"), Output("main-content", "style"), Output("operator-display", "children")],
    Input("login-btn", "n_clicks"), 
    [State("username", "value"), State("password", "value"), State("profession", "value")]
)
def auth(n, u, p, prof):
    if n > 0 and u == VALID_USER and p == VALID_PASS:
        display_text = f"OPERATOR: {u.upper()} | ROLE: {prof.upper() if prof else 'STAFF'}"
        return {"display": "none"}, {"display": "block"}, display_text
    
    # Login screen style
    login_style = {
        "position": "fixed", "top": 0, "left": 0, "width": "100%", "height": "100%", 
        "display": "flex", "alignItems": "center", "justifyContent": "center", 
        "zIndex": "1000", "background": "#001b2e"
    }
    return login_style, {"display": "none"}, ""

@app.callback(
    [Output("webcam-feed", "src"), Output("force-val", "children"), 
     Output("status-box", "children"), Output("status-box", "style")],
    Input("cam-timer", "n_intervals")
)
def update_wave(n):
    img, force, status = get_motion_metrics()
    col = ALERT_RED if status == "TURBULENT" else OK_GREEN
    style = {"color": col, "border": f"2px solid {col}", "padding": "15px", "fontWeight": "bold", 
             "fontSize": "22px", "backgroundColor": "rgba(0,0,0,0.6)", "textAlign":"center", "borderRadius":"10px"}
    return f"data:image/jpeg;base64,{img}" if img else "", f"WAVE FORCE: {force:.2f} N", f"WAVE: {status}", style

@app.callback(
    [Output(f"g-{k}", "figure") for k in ['temp', 'press', 'volt', 'curr']] +
    [Output(f"d-{k}", "children") for k in ['temp', 'press', 'volt', 'curr']] +
    [Output("voice-store", "data"), Output("solution-panel", "children"), Output("solution-panel", "style")],
    Input("timer", "n_intervals")
)
def update_telemetry(n):
    t, p, v, c = round(random.uniform(28, 36), 1), round(random.uniform(0.9, 1.4), 2), round(random.uniform(22, 24), 1), round(random.uniform(6, 10), 1)
    for k, val in zip(['temp', 'press', 'volt', 'curr'], [t, p, v, c]): history[k].append(val)
    
    v_msg, instr, col, action = "", "CORE STABLE", OK_GREEN, "No action required."
    if t > 33.5:
        v_msg = "Overheat detected."
        instr, col, action = "🔴 EMERGENCY: COOLING ACTIVE", ALERT_RED, "Initiate liquid nitrogen cooling."
    elif p > 1.25:
        v_msg = "Pressure warning."
        instr, col, action = "🟠 CAUTION: HIGH PRESSURE", "#ffa500", "Slowly ascend to 500m depth."

    mission_log_buffer.append({
        "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature_C": t, "Voltage_V": v, "Current_A": c, "Pressure_ATM": p,
        "Status": instr, "Solution": action
    })

    def fig(data, color):
        f = go.Figure(go.Scatter(y=list(data), fill='tozeroy', line=dict(color=color, width=4)))
        f.update_layout(height=100, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                         margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
        return f

    return (fig(history['temp'], CYAN_GLOW), fig(history['press'], "#00b4d8"),
            fig(history['volt'], ALERT_RED), fig(history['curr'], OK_GREEN),
            f"{t}°C", f"{p} ATM", f"{v} V", f"{c} A", v_msg, instr,
            {"borderLeft": f"15px solid {col}", "color": col, "background": "rgba(0,40,60,0.9)", "padding": "20px", "fontSize":"1.5rem", "borderRadius":"10px"})

@app.callback(Output("download-log-csv", "data"), Input("btn-download", "n_clicks"), prevent_initial_call=True)
def download_csv(n):
    df = pd.DataFrame(mission_log_buffer)
    return dcc.send_data_frame(df.to_csv, "DeepSea_Mission_Log.csv", index=False)

app.clientside_callback(
    "function(m){if(m){let s=new SpeechSynthesisUtterance(m); s.rate=0.85; window.speechSynthesis.speak(s);}return '';}",
    Output("voice-trigger", "children"), Input("voice-store", "data")
)

# --- CSS STYLING ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%} <title>DEEPSEA AI.CORE</title> {%favicon%} {%css%}
        <style>
            body { margin:0; background:#001b2e; font-family:'Courier New', monospace; color:white; overflow-x: hidden;}
            
            /* --- ANIMATION ELEMENTS --- */
            .ocean-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; }
            
            .sub-container { position: absolute; top: 10%; left: -150px; animation: patrol 30s infinite linear; font-size: 60px; }
            @keyframes patrol {
                0% { left: -100px; transform: scaleX(1); }
                48% { transform: scaleX(1); }
                50% { left: 105%; transform: scaleX(-1); }
                98% { transform: scaleX(-1); }
                100% { left: -100px; transform: scaleX(1); }
            }

            .fish { position: absolute; font-size: 30px; opacity: 0.5; z-index: 0; }
            @keyframes swim { from { left: -50px; } to { left: 105%; } }
            .f1 { top: 25%; animation: swim 20s infinite linear; }
            .f2 { top: 55%; animation: swim 25s infinite linear; animation-delay: 5s; }
            .f3 { top: 85%; animation: swim 15s infinite linear; animation-delay: 2s; }

            .bubble { position:absolute; bottom:-50px; background:rgba(255,255,255,0.1); border-radius:50%; animation:float 12s infinite linear; }
            @keyframes float { 0% {transform:translateY(0); opacity:0;} 50% {opacity:0.5;} 100% {transform:translateY(-110vh); opacity:0;} }

            /* --- UI COMPONENTS --- */
            .login-box { padding:40px; background:rgba(0,45,65,0.8); border:2px solid #00e5ff; border-radius:15px; width:400px; box-shadow:0 0 30px #00e5ff; z-index:1001; }
            .login-label { color: #00e5ff; font-size: 12px; font-weight: bold; margin-bottom: 5px; display: block; }
            .deep-input { display:block; margin-bottom:15px; padding:12px; background:#001b2e; color:#00e5ff; border:1px solid #00e5ff; width:100%; box-sizing: border-box; }
            .deep-dropdown .Select-control { background: #001b2e !important; border: 1px solid #00e5ff !important; margin-bottom: 15px; }
            .deep-dropdown .Select-placeholder, .deep-dropdown .Select-value-label { color: #00e5ff !important; }
            .glow-button { background:#00e5ff; border:none; padding:15px; width:100%; font-weight:bold; cursor:pointer; color:#001b2e; margin-top:10px; }
            
            .telemetry-row { display: flex; align-items: center; background: rgba(0, 50, 80, 0.7); border: 1px solid #00e5ff; border-radius: 10px; padding: 10px; gap: 20px; position: relative; z-index: 2; }
            .stat-card { min-width: 250px; text-align: left; padding-left: 10px; border-right: 1px solid rgba(0,229,255,0.3); }
            .val-glow { color:#00e5ff; text-shadow: 0 0 15px #00e5ff; font-size: 2.5rem !important; margin: 5px 0; }
            .stack-graph { flex: 1; height: 100px; }
            
            .webcam-frame { width:100%; border:3px solid #00e5ff; border-radius:10px; }
            .force-text { text-align:center; color:#00e5ff; font-size:1.8rem; font-weight:bold; margin:15px 0; }
            .nav-btn { background:#00e676; border:none; padding:12px 30px; border-radius:30px; font-weight:bold; cursor:pointer; margin-top:10px; color: #001b2e; }
            
            #main-content { position: relative; z-index: 5; }
        </style>
    </head>
    <body> {%app_entry%} <footer> {%config%} {%scripts%} {%renderer%} </footer> </body>
</html>
'''

if __name__ == "__main__":
    app.run(debug=True)