import os
import sys
import html
import math

import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# 1. INITIAL SETUP
# =========================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from database import db
from ml import ingest
from backend import search as srch
from ml import slm

st.set_page_config(
    page_title="EL Research Repository",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

db.init_db()


# =========================================================
# 2. DESIGN SYSTEM
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --orange: #E96D35;
    --orange-dark: #C95528;
    --orange-light: #FFF0E5;
    --ink: #26313D;
    --muted: #77818D;
    --border: #EAE8E2;
    --cream: #F8F7F3;
    --sage: #E5EDE5;
    --lavender: #EFEBFA;
    --blue: #E8F1FB;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 10%, rgba(225,235,225,0.45), transparent 35%),
        radial-gradient(ellipse at 95% 80%, rgba(249,230,215,0.42), transparent 32%),
        var(--cream);
    color: var(--ink);
}

.block-container {
    max-width: 1600px;
    padding-top: 4.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--ink);
    letter-spacing: -0.45px;
}

h1 {
    font-size: 34px !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 25px !important;
    font-weight: 650 !important;
}

h3 {
    font-size: 19px !important;
    font-weight: 650 !important;
}

p, label, input, button {
    font-family: 'DM Sans', sans-serif;
}

/* ---------------- SIDEBAR ---------------- */

[data-testid="stSidebar"] {
    background: #FFFEFC;
    border-right: 1px solid #ECE9E2;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] .stButton button {
    text-align: left;
    justify-content: flex-start;
    background: transparent;
    color: #46515F;
    border: 1px solid transparent;
    border-radius: 9px;
    font-size: 14px;
    font-weight: 600;
    padding: 10px 12px;
    min-height: 40px;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: #FFF0E5;
    color: var(--orange-dark);
}

[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: #FCE9DD;
    color: var(--orange-dark);
    border-left: 3px solid var(--orange);
}

/* ---------------- SECTION LABELS ---------------- */

.section-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    color: var(--orange-dark);
    text-transform: uppercase;
    margin-bottom: 7px;
}

/* ---------------- METRICS ---------------- */

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.94);
    border: 1px solid #EAE8E2;
    border-radius: 15px;
    padding: 17px 19px;
    box-shadow: 0 5px 18px rgba(49,55,47,0.035);
    min-height: 112px;
}

[data-testid="stMetricLabel"] {
    color: #64717D;
    font-size: 13px;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    color: var(--orange);
    font-family: 'Outfit', sans-serif;
    font-weight: 750;
}

/* ---------------- CHARTS ---------------- */

[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.94);
    border: 1px solid #ECEAE4;
    border-radius: 13px;
    padding: 6px;
}

/* ---------------- PROJECT EXPANDERS ---------------- */

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.94);
    border: 1px solid #EAE8E2;
    border-radius: 11px;
}

/* ---------------- INPUTS ---------------- */

.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stTextArea textarea {
    border-radius: 9px;
}

.stButton button {
    border-radius: 9px;
    font-weight: 600;
    min-height: 40px;
}

.stButton button[kind="primary"] {
    background: var(--orange);
    border-color: var(--orange);
}

.stButton button[kind="primary"]:hover {
    background: var(--orange-dark);
    border-color: var(--orange-dark);
}

hr {
    border-color: #E9E6DE;
}

/* ---------------- RECENT PROJECT CARDS ---------------- */

.recent-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(255,255,255,0.94);
    border: 1px solid #ECEAE4;
    border-radius: 11px;
    padding: 12px 13px;
    margin-bottom: 9px;
    transition: 0.2s ease;
}

.recent-card:hover {
    border-color: #E8B59B;
    transform: translateY(-1px);
}

.recent-icon {
    width: 38px;
    height: 38px;
    min-width: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
}

.recent-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 700;
    color: #303A46;
    line-height: 1.45;
}

.recent-meta {
    font-size: 10px;
    color: #89919A;
    margin-top: 4px;
}

/* ---------------- RESPONSIVE ---------------- */

@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATA FUNCTIONS
# =========================================================

def get_projects():
    return db.list_projects()


def get_domains():
    return db.list_domains()


def domain_options():
    return sorted(set(ingest.DOMAINS) | set(get_domains()))


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"

all_projects = get_projects()
domains = get_domains()

with st.sidebar:

    st.html("""
    <div style="padding:7px 0 18px 0;">
        <div style="
            font-family:'Outfit',sans-serif;
            font-size:24px;
            font-weight:800;
            color:#E96D35;
            letter-spacing:-0.7px;">
            EL REPOSITORY
        </div>
        <div style="
            font-size:12px;
            color:#77818D;
            margin-top:4px;">
            RV College of Engineering
        </div>
    </div>
    """)

    st.divider()
    st.caption("W O R K S P A C E")

    navigation = [
        ("Overview", "⌂   Overview"),
        ("Browse Projects", "▤   Browse Projects"),
        ("Intelligent Search", "⌕   Intelligent Search"),
        ("AI Assistant", "✦   AI Assistant"),
        ("Upload Documents", "⇧   Upload Documents")
    ]

    for page_name, label in navigation:

        if st.button(
            label,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if st.session_state.page == page_name else "secondary"
        ):
            st.session_state.page = page_name
            st.rerun()

    st.divider()
    st.caption("R E S E A R C H  &  I N S I G H T S")

    extra_pages = [
        ("Faculty Directory", "♧   Faculty Directory"),
        ("Research Analytics", "▥   Research Analytics"),
        ("SDG 9 Insights", "◇   SDG 9 Insights")
    ]

    for page_name, label in extra_pages:

        if st.button(
            label,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if st.session_state.page == page_name else "secondary"
        ):
            st.session_state.page = page_name
            st.rerun()

    st.divider()
    st.caption("R E P O S I T O R Y  S U M M A R Y")

    st.metric("Total Projects", len(all_projects))
    st.metric("Research Domains", len(domains))

    st.divider()

    st.html("""
    <div style="
        color:#E96D35;
        font-size:10px;
        font-weight:800;
        letter-spacing:1.8px;
        line-height:1.8;
        padding:5px 0;">
        DISCOVER · CONNECT<br>
        BUILD · INNOVATE
    </div>
    """)

page = st.session_state.page


# =========================================================
# 5. COMMON COMPONENTS
# =========================================================

def domain_filter(prefix):

    return st.selectbox(
        "Search / select domain",
        options=["All"] + get_domains(),
        index=0,
        key=f"{prefix}_domain",
        accept_new_options=True,
        placeholder="Type or select a domain..."
    )


def project_card(p, prefix):

    key = f"{prefix}_{p['id']}"

    with st.expander(
        f"📁  {p.get('title') or 'Untitled project'}  |  "
        f"{p.get('domain') or 'Unclassified'}  |  "
        f"{p.get('year') or 'Year N/A'}"
    ):

        st.write(
            f"**Team:** {p.get('team') or '-'}  |  "
            f"**Mentor:** {p.get('mentor') or '-'}"
        )

        if p.get("snippet"):
            st.caption("Matching content: " + p["snippet"])

        path = os.path.join(ingest.RAW, p["filename"])

        # file is read only when the user asks for it
        if os.path.exists(path):

            if st.button("Prepare download", key=f"prep_{key}"):

                with open(path, "rb") as f:
                    st.download_button(
                        "Download document",
                        data=f.read(),
                        file_name=p["filename"],
                        key=f"download_{key}"
                    )

        if st.checkbox("Edit project details", key=f"edit_{key}"):

            with st.form(f"form_{key}"):

                title_input = st.text_input(
                    "Project title",
                    p.get("title") or ""
                )

                year_input = st.text_input(
                    "Year",
                    p.get("year") or ""
                )

                team_input = st.text_input(
                    "Team",
                    p.get("team") or ""
                )

                mentor_input = st.text_input(
                    "Mentor",
                    p.get("mentor") or ""
                )

                options = ["Auto-detect"] + domain_options()
                current = p.get("domain_override") or ""

                selected_domain = st.selectbox(
                    "Domain",
                    options,
                    index=options.index(current) if current in options else 0,
                    accept_new_options=True
                )

                if st.form_submit_button("Save changes"):

                    final_domain = (
                        "" if selected_domain == "Auto-detect"
                        else selected_domain
                    )

                    db.update_project(
                        p["id"],
                        title=title_input,
                        year=year_input,
                        team=team_input,
                        mentor=mentor_input,
                        domain_override=final_domain
                    )

                    st.success("Project details updated.")
                    st.rerun()

        if st.checkbox("Delete project", key=f"delete_{key}"):

            if st.button("Confirm deletion", key=f"confirm_{key}"):
                ingest.remove_document(p["id"])
                # drop cached search results so the deleted project disappears
                st.session_state.pop("results", None)
                st.rerun()


# =========================================================
# 6. ILLUSTRATED CAMPUS HERO
# =========================================================

def show_hero():

    st.html("""
    <div style="
        position:relative;
        width:100%;
        min-height:300px;
        box-sizing:border-box;
        overflow:hidden;
        border-radius:18px;
        background:
            linear-gradient(90deg,
                rgba(250,247,239,0.98) 0%,
                rgba(250,247,239,0.91) 37%,
                rgba(250,247,239,0.25) 100%),
            linear-gradient(180deg,#E8EDEB,#F5E7D8);
        border:1px solid rgba(220,215,202,0.8);
        box-shadow:0 10px 28px rgba(63,69,61,0.07);
        font-family:'DM Sans',sans-serif;">

        <!-- Hand-drawn campus illustration -->
        <svg viewBox="0 0 1200 360"
             preserveAspectRatio="xMidYMid slice"
             style="
                position:absolute;
                right:0;
                bottom:0;
                width:76%;
                height:100%;
                opacity:0.84;
                pointer-events:none;">

            <defs>
                <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#E4EBE9"/>
                    <stop offset="100%" stop-color="#F7E8D9"/>
                </linearGradient>
            </defs>

            <path d="M0 270 Q220 225 430 260 T850 240 T1200 260"
                  fill="none" stroke="#A8B7AA" stroke-width="2"/>

            <!-- distant buildings -->
            <g fill="#D7DCD4" stroke="#899A91" stroke-width="2"
               stroke-linejoin="round">
                <path d="M160 255 L160 160 L235 130 L310 160 L310 255 Z"/>
                <path d="M310 255 L310 180 L375 145 L440 180 L440 255 Z"/>
                <path d="M440 255 L440 195 L500 170 L560 195 L560 255 Z"/>
            </g>

            <!-- distant windows -->
            <g fill="#F5E4CB" stroke="#8D9B92" stroke-width="1.5">
                <rect x="180" y="177" width="18" height="22"/>
                <rect x="215" y="165" width="18" height="22"/>
                <rect x="252" y="177" width="18" height="22"/>
                <rect x="180" y="215" width="18" height="22"/>
                <rect x="215" y="215" width="18" height="22"/>
                <rect x="252" y="215" width="18" height="22"/>

                <rect x="332" y="190" width="19" height="22"/>
                <rect x="369" y="172" width="19" height="22"/>
                <rect x="406" y="190" width="19" height="22"/>
            </g>

            <!-- main college building -->
            <g stroke="#677D79" stroke-width="3"
               stroke-linejoin="round" stroke-linecap="round">

                <path d="M570 257 L570 128 L735 72 L920 125 L920 257 Z"
                      fill="#F1E8D8"/>

                <path d="M570 128 L735 72 L920 125"
                      fill="none" stroke-width="4"/>

                <path d="M600 136 L735 91 L890 133"
                      fill="none" stroke="#B58F6D" stroke-width="2"/>

                <!-- roof -->
                <path d="M555 132 L735 58 L937 128"
                      fill="#D9C6AC"/>

                <!-- columns -->
                <path d="M605 150 L605 255 M645 137 L645 255
                         M685 124 L685 255 M725 112 L725 255
                         M765 112 L765 255 M805 123 L805 255
                         M845 135 L845 255 M885 145 L885 255"
                      fill="none" stroke-width="2"/>

                <!-- windows -->
                <g fill="#DCE7E3" stroke-width="1.7">
                    <rect x="615" y="165" width="23" height="36"/>
                    <rect x="654" y="153" width="23" height="36"/>
                    <rect x="694" y="141" width="23" height="36"/>
                    <rect x="734" y="132" width="23" height="36"/>
                    <rect x="774" y="141" width="23" height="36"/>
                    <rect x="814" y="153" width="23" height="36"/>
                    <rect x="854" y="165" width="23" height="36"/>

                    <rect x="615" y="211" width="23" height="31"/>
                    <rect x="654" y="211" width="23" height="31"/>
                    <rect x="694" y="211" width="23" height="31"/>
                    <rect x="734" y="211" width="23" height="31"/>
                    <rect x="774" y="211" width="23" height="31"/>
                    <rect x="814" y="211" width="23" height="31"/>
                    <rect x="854" y="211" width="23" height="31"/>
                </g>
            </g>

            <!-- College sign -->
            <g transform="rotate(-10 735 118)">
                <rect x="665" y="95" width="145" height="25"
                      rx="3" fill="#F8F4E9"
                      stroke="#8B8171" stroke-width="1.5"/>
                <text x="737" y="112"
                      font-size="11"
                      font-weight="700"
                      text-anchor="middle"
                      fill="#53635F">
                    RV COLLEGE OF ENGINEERING
                </text>
            </g>

            <!-- trees -->
            <g fill="none" stroke="#718B75" stroke-width="3"
               stroke-linecap="round">
                <path d="M1000 280 Q1004 205 1018 145"/>
                <path d="M1018 210 Q980 175 970 145"/>
                <path d="M1015 190 Q1050 160 1060 125"/>
                <path d="M1065 285 Q1072 220 1090 180"/>
                <path d="M1080 225 Q1115 195 1120 165"/>
            </g>

            <g fill="#B7C7AC" stroke="#7D967E" stroke-width="1.5">
                <ellipse cx="975" cy="145" rx="40" ry="22"/>
                <ellipse cx="1015" cy="130" rx="42" ry="28"/>
                <ellipse cx="1050" cy="151" rx="37" ry="24"/>
                <ellipse cx="1085" cy="180" rx="40" ry="25"/>
                <ellipse cx="1120" cy="165" rx="35" ry="23"/>
            </g>

            <!-- books on the right -->
            <g stroke="#786B5F" stroke-width="2" stroke-linejoin="round">
                <path d="M960 278 L1145 278 L1145 303 L960 303 Z"
                      fill="#C77E55"/>
                <path d="M975 246 L1155 246 L1155 275 L975 275 Z"
                      fill="#E9D9BD"/>
                <path d="M990 214 L1165 214 L1165 243 L990 243 Z"
                      fill="#829B8B"/>
                <path d="M1008 184 L1175 184 L1175 211 L1008 211 Z"
                      fill="#D8B18B"/>
            </g>

            <g fill="#F8F3E8" font-size="15"
               font-family="Georgia,serif" font-style="italic">
                <text x="1040" y="296">Research</text>
                <text x="1050" y="266">Innovation</text>
                <text x="1055" y="234">Knowledge</text>
                <text x="1060" y="203">Learning</text>
            </g>

            <!-- ground and sketch details -->
            <path d="M0 305 Q260 285 500 302 T1000 294 T1200 300"
                  fill="none" stroke="#9AAB9C" stroke-width="2"/>

            <g fill="none" stroke="#A5B5A5" stroke-width="1.5"
               stroke-linecap="round">
                <path d="M65 290 Q90 265 110 290"/>
                <path d="M115 300 Q135 270 155 300"/>
                <path d="M485 280 Q510 260 530 280"/>
                <path d="M920 300 Q950 278 970 300"/>
                <path d="M112 320 L210 320"/>
                <path d="M360 315 L430 315"/>
                <path d="M790 315 L870 315"/>
            </g>
        </svg>

        <!-- Text area -->
        <div style="
            position:relative;
            z-index:2;
            width:58%;
            padding:34px 36px 32px 36px;">

            <div style="
                display:inline-block;
                background:#E96D35;
                color:white;
                padding:7px 13px;
                border-radius:20px;
                font-size:10px;
                font-weight:800;
                letter-spacing:1.2px;
                margin-bottom:18px;">
                SDG 9 · INDUSTRY, INNOVATION AND INFRASTRUCTURE
            </div>

            <div style="
                font-family:'Outfit',sans-serif;
                font-size:clamp(27px,3vw,39px);
                line-height:1.12;
                font-weight:750;
                color:#26313D;
                letter-spacing:-1px;
                margin-bottom:14px;">
                Intelligent EL<br>
                <span style="color:#E96D35;">
                    Research Repository
                </span>
            </div>

            <div style="
                color:#5E6872;
                font-size:14px;
                line-height:1.8;
                max-width:510px;">
                A centralized research space to discover previous work,
                connect expertise and build future innovation.
            </div>

            <div style="
                margin-top:17px;
                font-size:10px;
                font-weight:700;
                color:#B16C4A;
                letter-spacing:1.5px;">
                DISCOVER · CONNECT · BUILD · INNOVATE
            </div>
        </div>
    </div>
    """)


# =========================================================
# 7. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    show_hero()

    # Statistics row

    mentor_count = len(set(
        p.get("mentor") for p in all_projects if p.get("mentor")
    ))

    year_count = len(set(
        p.get("year") for p in all_projects if p.get("year")
    ))

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📁  Total Projects", len(all_projects))
    c2.metric("▱  Research Domains", len(domains))
    c3.metric("♧  Faculty Mentors", mentor_count)
    c4.metric("▦  Academic Years", year_count)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Two-column dashboard

    left, right = st.columns([1.8, 1], gap="medium")

    with left:

        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.94);
            border:1px solid #ECEAE4;
            border-radius:15px;
            padding:17px 19px 4px 19px;
            margin-bottom:12px;">
            <div style="
                font-family:'Outfit',sans-serif;
                font-size:19px;
                font-weight:700;
                color:#29333E;">
                Research Distribution
            </div>
            <div style="font-size:11px;color:#89919A;margin-top:4px;">
                An overview of projects across different research domains.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if all_projects:

            domain_data = pd.DataFrame([
                {"Domain": p.get("domain") or "Unclassified"}
                for p in all_projects
            ])

            distribution = (
                domain_data["Domain"]
                .value_counts()
                .reset_index()
            )

            distribution.columns = ["Domain", "Projects"]

                        chart1 = st.container()
            chart2 = st.container()

            with chart1:

                short = distribution.copy()
                short["Domain"] = (
                    short["Domain"]
                    .str.replace(" / ", "/")
                    .str.replace(" & ", " &<br>")
                )

                fig = px.bar(
                    short,
                    x="Domain",
                    y="Projects",
                    title="Projects Across Research Domains",
                    color_discrete_sequence=["#F39A63"]
                )

                fig.update_layout(
                    height=380,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(
                        family="DM Sans, sans-serif",
                        color="#56616B",
                        size=12
                    ),
                    title_font=dict(
                        family="Outfit, sans-serif",
                        size=16,
                        color="#303A46"
                    ),
                    xaxis_title=None,
                    yaxis_title="Number of projects",
                    showlegend=False,
                    margin=dict(l=60, r=20, t=55, b=70)
                )

                fig.update_xaxes(tickangle=0, automargin=True)
                fig.update_yaxes(
                    rangemode="tozero",
                    gridcolor="#EFEDE8",
                    automargin=True
                )
                fig.update_traces(marker_line_width=0)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    theme=None,
                    config={"displayModeBar": False}
                )

            with chart2:

                st.markdown(
                    "<div style='height:10px'></div>",
                    unsafe_allow_html=True
                )

                fig2 = px.pie(
                    distribution,
                    names="Domain",
                    values="Projects",
                    hole=0.6,
                    title="Domain-wise Project Share",
                    color_discrete_sequence=[
                        "#E97843",
                        "#F4B58E",
                        "#A9BBAA",
                        "#D7C5A6",
                        "#B9A9D6",
                        "#91B8D0",
                        "#D8A5A5"
                    ]
                )

                fig2.update_layout(
                    height=400,
                    paper_bgcolor="white",
                    font=dict(
                        family="DM Sans, sans-serif",
                        color="#56616B",
                        size=12
                    ),
                    title_font=dict(
                        family="Outfit, sans-serif",
                        size=16,
                        color="#303A46"
                    ),
                    margin=dict(l=10, r=10, t=55, b=20),
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        x=1.02,
                        font=dict(size=12)
                    )
                )

                fig2.update_traces(
                    textposition="inside",
                    textinfo="percent",
                    insidetextorientation="horizontal"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True,
                    theme=None,
                    config={"displayModeBar": False}
                )


            st.info("Upload projects to see research distribution.")

    with right:

        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.94);
            border:1px solid #ECEAE4;
            border-radius:15px;
            padding:17px 19px;
            margin-bottom:12px;">
            <div style="
                font-family:'Outfit',sans-serif;
                font-size:19px;
                font-weight:700;
                color:#29333E;">
                Recently Added Projects
            </div>
            <div style="font-size:11px;color:#89919A;margin-top:4px;">
                The latest research projects uploaded to the repository.
            </div>
        </div>
        """, unsafe_allow_html=True)

        recent = sorted(
            all_projects,
            key=lambda p: p.get("uploaded_at") or "",
            reverse=True
        )[:5]

        card_colours = [
            ("#FCE5D8", "#D96F43", "▤"),
            ("#E1EFE2", "#56876B", "▣"),
            ("#ECE7FA", "#8063B7", "▤"),
            ("#E1EEFA", "#508FC2", "▧"),
            ("#F8E4E8", "#C66B83", "▣")
        ]

        if recent:

            for i, p in enumerate(recent):

                title = html.escape(
                    str(p.get("title") or "Untitled project")
                )

                domain = html.escape(
                    str(p.get("domain") or "Unclassified")
                )

                year = html.escape(
                    str(p.get("year") or "Year not specified")
                )

                bg, fg, symbol = card_colours[i % len(card_colours)]

                st.html(f"""
                <div class="recent-card">
                    <div class="recent-icon"
                         style="background:{bg};color:{fg};">
                        {symbol}
                    </div>
                    <div style="min-width:0;flex:1;">
                        <div class="recent-title">
                            {title}
                        </div>
                        <div class="recent-meta">
                            {domain} &nbsp; | &nbsp; {year}
                        </div>
                    </div>
                    <div style="color:#8D969F;font-size:18px;">
                        ›
                    </div>
                </div>
                """)

        else:

            st.info("No projects have been uploaded yet.")


# =========================================================
# 8. BROWSE PROJECTS
# =========================================================

elif page == "Browse Projects":

    st.markdown(
        '<div class="section-label">EXPLORE EXISTING WORK</div>',
        unsafe_allow_html=True
    )

    st.title("Project Repository")

    st.caption(
        "Browse, filter and explore previously completed EL projects."
    )

    st.divider()

    selected_domain = domain_filter("browse")

    query = st.text_input(
        "Search projects",
        placeholder="Enter project title, team or mentor..."
    )

    items = [
        p for p in db.list_projects(search=query or None)
        if srch.dom_ok(p, selected_domain)
    ]

    st.write(f"**{len(items)} projects found**")

    pages = max(1, math.ceil(len(items) / 10))

    current_page = st.number_input(
        "Page",
        min_value=1,
        max_value=pages,
        value=1
    )

    for p in items[(current_page - 1) * 10: current_page * 10]:
        project_card(p, "browse")


# =========================================================
# 9. INTELLIGENT SEARCH
# =========================================================

elif page == "Intelligent Search":

    st.markdown(
        '<div class="section-label">AI-POWERED DISCOVERY</div>',
        unsafe_allow_html=True
    )

    st.title("Intelligent Research Search")

    st.caption(
        "Discover relevant projects through keyword and semantic retrieval."
    )

    st.divider()

    query = st.text_input(
        "What are you researching?",
        placeholder="e.g. Energy-efficient manufacturing using IoT"
    )

    c1, c2 = st.columns(2)

    mode = c1.selectbox(
        "Search method",
        ["hybrid", "semantic", "keyword"],
        help="Keyword finds exact words. Semantic finds similar meaning. Hybrid combines both."
    )

    max_results = c2.slider(
        "Maximum results",
        3,
        20,
        5
    )

    selected_domain = domain_filter("search")

    # results are stored in session_state so they survive
    # clicks on Edit / Delete / Prepare download inside the cards
    if st.button(
        "Search Repository",
        type="primary",
        use_container_width=True
    ):

        if query.strip():

            with st.spinner("Searching the repository..."):

                st.session_state.results = srch.search(
                    query,
                    k=max_results,
                    domain=selected_domain,
                    mode=mode
                )

        else:

            st.warning("Please enter a search query.")

    results = st.session_state.get("results")

    if results is not None:

        st.divider()

        st.subheader(f"{len(results)} relevant projects found")

        if results:

            for p in results:
                project_card(p, "search")

        else:

            st.warning(
                "No matching projects found. Try different search terms."
            )


# =========================================================
# 10. AI ASSISTANT
# =========================================================

elif page == "AI Assistant":

    st.markdown(
        '<div class="section-label">SLM-POWERED ASSISTANT</div>',
        unsafe_allow_html=True
    )

    st.title("EL Research Assistant")

    st.caption(
        "Ask in plain language. Answers come only from the repository's documents."
    )

    st.divider()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    if "ai_mode" not in st.session_state:
        st.session_state.ai_mode = None

    c1, c2, c3 = st.columns([1, 1, 1])

    if c1.button(
        "Find similar projects",
        use_container_width=True,
        type="primary" if st.session_state.ai_mode == "similar" else "secondary"
    ):
        st.session_state.ai_mode = (
            None if st.session_state.ai_mode == "similar" else "similar"
        )
        st.rerun()

    if c2.button(
        "Suggest research gaps",
        use_container_width=True,
        type="primary" if st.session_state.ai_mode == "gaps" else "secondary"
    ):
        st.session_state.ai_mode = (
            None if st.session_state.ai_mode == "gaps" else "gaps"
        )
        st.rerun()

    if c3.button("Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.session_state.ai_mode = None
        st.rerun()

    if st.session_state.ai_mode == "similar":
        st.info("Mode: Find similar projects. Type a topic or a project title below.")
    elif st.session_state.ai_mode == "gaps":
        st.info("Mode: Suggest research gaps. Type a topic below.")

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources"):
                    for s in m["sources"]:
                        st.write(f"• {s}")

    user_q = st.chat_input("Ask about past EL projects...")

    if user_q:

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat
        ]

        st.session_state.chat.append({"role": "user", "content": user_q})

        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):

            with st.spinner("Searching the repository..."):
                token_stream, hits = slm.ask(
                    user_q, history, task=st.session_state.ai_mode
                )

            answer = st.write_stream(token_stream)

            sources = []

            for h in hits:
                p = h["project"]
                line = f"{p['title']}  ({p['domain']}, {p['year'] or 'year n/a'})"
                if line not in sources:
                    sources.append(line)

            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"• {s}")

        st.session_state.chat.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


# =========================================================
# 11. UPLOAD DOCUMENTS
# =========================================================

elif page == "Upload Documents":

    st.markdown(
        '<div class="section-label">CONTRIBUTE TO THE REPOSITORY</div>',
        unsafe_allow_html=True
    )

    st.title("Upload Research Documents")

    st.caption(
        "Add EL project reports and research documents "
        "to the centralized repository."
    )

    st.divider()

    st.info("Supported formats: PDF and DOCX")

    files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    st.subheader("Project Information")

    c1, c2 = st.columns(2)

    team = c1.text_input("Team name (optional)")
    mentor = c2.text_input("Faculty mentor (optional)")

    c1, c2 = st.columns(2)

    year = c1.text_input("Academic year (optional)")

    selected_domain = c2.selectbox(
        "Domain",
        ["Auto-detect"] + domain_options(),
        accept_new_options=True,
        placeholder="Type or select a domain..."
    )

    final_domain = (
        "" if selected_domain == "Auto-detect"
        else selected_domain
    )

    if st.button(
        "Process and Add Documents",
        type="primary",
        disabled=not files,
        use_container_width=True
    ):

        progress = st.progress(0.0)

        for i, file in enumerate(files, 1):

            with st.spinner(f"Processing {file.name}..."):

                success, message = ingest.add_document(
                    os.path.basename(file.name),
                    file.getvalue(),
                    team=team,
                    mentor=mentor,
                    year=year,
                    domain_override=final_domain
                )

            if success:
                st.success(message)
            else:
                st.error(message)

            progress.progress(i / len(files))

        # no st.rerun() here, so the messages above stay visible
        st.info("Done. Open Browse Projects to see the new files.")


# =========================================================
# 12. FUTURE MODULES
# =========================================================

elif page in ["Faculty Directory", "Research Analytics", "SDG 9 Insights"]:

    from analytics import pages as ap

    if page == "Research Analytics":
        ap.show_analytics(all_projects)
    elif page == "Faculty Directory":
        ap.show_faculty(all_projects)
    else:
        ap.show_sdg9(all_projects)
