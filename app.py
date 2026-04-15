"""
🏟️ BAYER LEVERKUSEN — TACTICAL BUILD-UP INTELLIGENCE DASHBOARD
Bundesliga 2023/24 | StatsBomb Event Data
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Leverkusen Tactical Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean coach-friendly UI
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    h1 { color: #E63946; }
    h2 { color: #264653; border-bottom: 2px solid #E63946; padding-bottom: 0.3rem; }
    h3 { color: #457B9D; }
    .stMetric { background: #f8f9fa; border-radius: 10px; padding: 10px; border-left: 4px solid #E63946; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; white-space: nowrap; overflow: visible; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; white-space: nowrap; overflow: visible; }
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #2A9D8F;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.95rem;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #E9C46A;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COLOR PALETTE (consistent)
# ─────────────────────────────────────────────────────────────
C_RED = "#E63946"        # aggressive / progression
C_BLUE = "#457B9D"       # structure
C_GREEN = "#2A9D8F"      # efficiency
C_DARK = "#264653"       # base / dark
C_YELLOW = "#E9C46A"     # warning / highlight
C_ORANGE = "#F4A261"     # accent
C_CORAL = "#E76F51"      # secondary negative

PALETTE = [C_RED, C_BLUE, C_GREEN, C_YELLOW, C_ORANGE, C_CORAL, C_DARK]


# ─────────────────────────────────────────────────────────────
# SAFE UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
def safe_x(loc):
    return loc[0] if isinstance(loc, list) and len(loc) > 0 else np.nan

def safe_y(loc):
    return loc[1] if isinstance(loc, list) and len(loc) > 1 else np.nan

def is_progressive(row):
    loc = row.get("location", None)
    end = row.get("pass_end_location", None)
    if isinstance(loc, list) and isinstance(end, list):
        if len(loc) > 0 and len(end) > 0:
            return (end[0] - loc[0]) >= 10
    return False

def insight_box(text):
    st.markdown(f'<div class="insight-box">💡 <b>Tactical Insight:</b> {text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading StatsBomb data… This may take a minute on first run.")
def load_data(comp_id, season_id, team_name):
    from statsbombpy import sb

    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    all_teams = sorted(pd.unique(matches[["home_team", "away_team"]].values.ravel()))

    team_matches = matches[
        (matches["home_team"] == team_name) | (matches["away_team"] == team_name)
    ]

    all_events = []
    for mid in team_matches["match_id"]:
        ev = sb.events(match_id=mid)
        ev["match_id"] = mid
        all_events.append(ev)

    events = pd.concat(all_events, ignore_index=True)
    return matches, team_matches, events, all_teams


@st.cache_data(show_spinner="Computing build-up metrics…")
def compute_buildup(_events, team_name):
    """Compute all build-up metrics from raw events."""
    team_ev = _events[_events["team"] == team_name].copy()
    team_ev = team_ev.sort_values(["match_id", "period", "timestamp"])
    team_ev["possession_id"] = team_ev["match_id"].astype(str) + "_" + team_ev["possession"].astype(str)

    build_types = ["Pass", "Carry", "Ball Receipt*"]
    build = team_ev[team_ev["type"].isin(build_types)].copy()
    build = build.groupby("possession_id").filter(lambda x: len(x) >= 3)

    build["x"] = build["location"].apply(safe_x)
    build["y"] = build["location"].apply(safe_y)
    build = build.dropna(subset=["x", "y"])
    build["zone"] = np.where(build["y"] < 26.6, "Left",
                     np.where(build["y"] > 53.3, "Right", "Center"))

    # PASSES
    passes = build[build["type"] == "Pass"].copy()
    passes["outcome"] = passes["pass_outcome"].fillna("Complete")
    passes["completed"] = passes["outcome"] == "Complete"
    passes["progressive"] = passes.apply(is_progressive, axis=1)
    passes["x"] = passes["location"].apply(safe_x)
    passes["y"] = passes["location"].apply(safe_y)

    # CARRIES
    carries = build[build["type"] == "Carry"].copy()
    carries["start_x"] = carries["location"].apply(safe_x)
    if "end_location" in carries.columns:
        carries["end_x"] = carries["end_location"].apply(safe_x)
    else:
        carries["end_x"] = np.nan
    carries["progression"] = carries["end_x"] - carries["start_x"]
    carries = carries.dropna(subset=["progression"])

    return build, passes, carries


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/df/Bundesliga_logo_%282017%29.svg/200px-Bundesliga_logo_%282017%29.svg.png", width=120)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/59/Bayer_04_Leverkusen_logo.svg/180px-Bayer_04_Leverkusen_logo.svg.png", width=80)
st.sidebar.title("⚽ Tactical Dashboard")
st.sidebar.markdown("**Bundesliga 2023/24**  \nStatsBomb Open Data")

COMP_ID = 9
SEASON_ID = 281
DEFAULT_TEAM = "Bayer Leverkusen"

# Load data once
try:
    matches, team_matches, events, all_teams = load_data(COMP_ID, SEASON_ID, DEFAULT_TEAM)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Team selector
selected_team = st.sidebar.selectbox(
    "🔄 Select Team",
    all_teams,
    index=all_teams.index(DEFAULT_TEAM) if DEFAULT_TEAM in all_teams else 0,
)

# Reload if team changed
if selected_team != DEFAULT_TEAM:
    try:
        matches, team_matches, events, all_teams = load_data(COMP_ID, SEASON_ID, selected_team)
    except Exception as e:
        st.error(f"Failed to load data for {selected_team}: {e}")
        st.stop()

# Match filter
match_options = {}
for _, row in team_matches.iterrows():
    mid = row["match_id"]
    opp = row["away_team"] if row["home_team"] == selected_team else row["home_team"]
    date = str(row.get("match_date", ""))[:10]
    match_options[f"vs {opp} ({date})"] = mid

selected_matches = st.sidebar.multiselect(
    "📅 Filter Matches",
    list(match_options.keys()),
    default=list(match_options.keys()),
    help="Select specific matches to analyse",
)

selected_match_ids = [match_options[m] for m in selected_matches]

# Filter events to selected matches
if selected_match_ids:
    filtered_events = events[events["match_id"].isin(selected_match_ids)]
else:
    filtered_events = events

# Compute build-up
build, passes, carries = compute_buildup(filtered_events, selected_team)

# Navigation
st.sidebar.markdown("---")
section = st.sidebar.radio(
    "📊 Dashboard Section",
    [
        "1. Overview",
        "2. Build-up Structure",
        "3. Progression Analysis",
        "4. Final Third & Chance Creation",
        "5. Player Impact",
        "6. Benchmark Comparison",
        "7. Match Preparation",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Analysing **{len(selected_matches)}** matches for **{selected_team}**")
st.sidebar.caption(f"Build-up actions: **{len(build):,}** | Passes: **{len(passes):,}**")


# ─────────────────────────────────────────────────────────────
# PRECOMPUTE COMMON METRICS
# ─────────────────────────────────────────────────────────────
total_sequences = build["possession_id"].nunique()
avg_seq_len = build.groupby("possession_id").size().mean() if total_sequences > 0 else 0
pass_completion = passes["completed"].mean() * 100 if len(passes) > 0 else 0

progressive_passes = passes[passes["progressive"]]
line_breaking = passes[passes["pass_length"].fillna(0) > 15]
final_third = passes[passes["x"] > 80]
z14_passes = passes[(passes["x"] > 102) & (passes["y"] > 18) & (passes["y"] < 62)]
crosses = passes[passes.get("pass_cross", pd.Series(dtype=bool)).fillna(False) == True] if "pass_cross" in passes.columns else pd.DataFrame()
long_passes = passes[passes["pass_length"].fillna(0) > 30]
shot_assists = passes[passes["pass_shot_assist"].fillna(False) == True] if "pass_shot_assist" in passes.columns else pd.DataFrame()

if "pass_technique" in passes.columns:
    through_balls = passes[passes["pass_technique"] == "Through Ball"]
else:
    through_balls = passes[passes["pass_goal_assist"].fillna(False) == True] if "pass_goal_assist" in passes.columns else pd.DataFrame()

progressive_carries = carries[carries["progression"] > 5]
ball_prog_index = len(progressive_passes) + len(progressive_carries)

prog_rate = len(progressive_passes) / len(passes) * 100 if len(passes) > 0 else 0

zone_dist = build["zone"].value_counts(normalize=True) * 100
zone_counts = build["zone"].value_counts()


# ─────────────────────────────────────────────────────────────
# HELPER: draw pitch on matplotlib ax
# ─────────────────────────────────────────────────────────────
def draw_pitch(ax, bg="#1a472a"):
    ax.set_xlim(-8, 128)
    ax.set_ylim(-8, 88)
    ax.set_facecolor(bg)
    lines = [
        ([0, 120, 120, 0, 0], [0, 0, 80, 80, 0]),
        ([60, 60], [0, 80]),
        ([0, 18, 18, 0], [18, 18, 62, 62]),
        ([120, 102, 102, 120], [18, 18, 62, 62]),
        ([0, 6, 6, 0], [30, 30, 50, 50]),
        ([120, 114, 114, 120], [30, 30, 50, 50]),
    ]
    for xs, ys in lines:
        ax.plot(xs, ys, color="white", linewidth=1.5, alpha=0.7)
    circle = plt.Circle((60, 40), 9.15, color="white", fill=False, linewidth=1.5, alpha=0.7)
    ax.add_patch(circle)
    ax.plot(60, 40, "o", color="white", markersize=3, alpha=0.7)
    ax.set_xlabel("Pitch Length", fontsize=10, color="white")
    ax.set_ylabel("Pitch Width", fontsize=10, color="white")
    ax.tick_params(colors="white")


def add_direction_arrows(ax):
    ax.annotate("", xy=(3, -5), xytext=(30, -5),
                arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.4", color="cyan", lw=3))
    ax.text(16, -7, "← DEFENSIVE", fontsize=9, color="cyan", ha="center", fontweight="bold")
    ax.annotate("", xy=(117, -5), xytext=(90, -5),
                arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.4", color="#FF4444", lw=3))
    ax.text(104, -7, "ATTACKING →", fontsize=9, color="#FF4444", ha="center", fontweight="bold")


# ═══════════════════════════════════════════════════════════════
# SECTION 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if section == "1. Overview":
    st.title(f"🏟️ {selected_team} — Tactical Build-up Overview")
    st.markdown(f"**Bundesliga 2023/24** · {len(selected_matches)} matches · StatsBomb Open Data")

    # KPI Metrics Row — two rows of 3 for readability
    c1, c2, c3 = st.columns(3)
    c1.metric("Build-up Sequences", f"{total_sequences:,}")
    c2.metric("Avg Sequence Length", f"{avg_seq_len:.1f}")
    c3.metric("Pass Completion %", f"{pass_completion:.1f}%")
    c4, c5, c6 = st.columns(3)
    c4.metric("Progressive Passes", f"{len(progressive_passes):,}")
    c5.metric("Progression Rate", f"{prog_rate:.1f}%")
    c6.metric("Ball Prog. Index", f"{ball_prog_index:,}")

    st.markdown("---")

    # KPI Bar Chart
    kpi_names = ["Progressive\nPasses", "Line-Breaking\nPasses", "Final Third\nPasses", "Zone 14\nEntries", "Shot\nAssists"]
    kpi_vals = [len(progressive_passes), len(line_breaking), len(final_third), len(z14_passes), len(shot_assists)]
    kpi_colors = [C_RED, C_BLUE, C_GREEN, C_YELLOW, C_ORANGE]

    fig_kpi = go.Figure()
    fig_kpi.add_trace(go.Bar(
        x=kpi_names, y=kpi_vals,
        marker_color=kpi_colors,
        text=kpi_vals,
        textposition="outside",
        textfont=dict(size=14, color="black"),
    ))
    fig_kpi.update_layout(
        title=dict(text="Key Performance Indicators — Offensive Build-up", font=dict(size=18)),
        yaxis_title="Count",
        xaxis_title="KPI Category",
        height=450,
        showlegend=False,
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_kpi, width="stretch")

    # Tactical interpretation
    dominant_zone = zone_dist.idxmax() if len(zone_dist) > 0 else "N/A"
    if dominant_zone == "Center":
        struct_note = "Central build-up preference — indicative of a possession-based system through midfield."
    elif dominant_zone == "Left":
        struct_note = "Left-sided build-up bias — suggests overloading through the left channel."
    else:
        struct_note = "Right-sided build-up emphasis — right flank used as primary progression corridor."

    if prog_rate > 25:
        prog_note = "High progression rate — vertical, direct approach to move the ball upfield."
    elif prog_rate > 15:
        prog_note = "Moderate progression rate — balanced between patience and directness."
    else:
        prog_note = "Low progression rate — patient, possession-recycling build-up style."

    insight_box(f"**Structure:** {struct_note}<br>**Progression:** {prog_note}<br>"
                f"**Pass Accuracy:** {'Elite-level accuracy' if pass_completion > 80 else 'Solid accuracy'} at {pass_completion:.1f}%.")

    # 6-panel mega dashboard (matplotlib)
    st.markdown("### 📋 Mega Dashboard — All Metrics at a Glance")

    fig_mega = plt.figure(figsize=(20, 14))
    fig_mega.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # Panel 1: KPI bars
    ax1 = fig_mega.add_subplot(gs[0, 0])
    bars = ax1.bar(kpi_names, kpi_vals, color=kpi_colors, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, kpi_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(kpi_vals)*0.02,
                 str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_title("KPI OVERVIEW", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Count"); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: Zone pie
    ax2 = fig_mega.add_subplot(gs[0, 1])
    if len(zone_counts) > 0:
        wedges, texts, autotexts = ax2.pie(
            zone_counts.values, labels=zone_counts.index,
            autopct=lambda p: f"{p:.1f}%\n({int(round(p/100*sum(zone_counts.values)))})",
            colors=[C_DARK, C_GREEN, C_YELLOW], startangle=140,
            textprops={"fontsize": 11}, wedgeprops={"edgecolor": "white", "linewidth": 2})
        for at in autotexts:
            at.set_fontweight("bold")
    ax2.set_title("ZONE DOMINANCE (L/C/R)", fontsize=13, fontweight="bold")

    # Panel 3: Top 10 players
    ax3 = fig_mega.add_subplot(gs[1, 0])
    _top10 = build["player"].value_counts().head(10)
    pn = _top10.index[::-1].tolist(); pv = _top10.values[::-1].tolist()
    bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(pn)))
    bars3 = ax3.barh(pn, pv, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars3, pv):
        ax3.text(bar.get_width() + max(pv)*0.02, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontsize=10, fontweight="bold")
    ax3.set_title("TOP 10 BUILD-UP PLAYERS", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Actions"); ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel 4: Chance creation
    ax4 = fig_mega.add_subplot(gs[1, 1])
    cc_names = ["Crosses", "Long Passes", "Short Assists", "Through Balls"]
    _short_a = len(passes[(passes.get("pass_shot_assist", pd.Series(dtype=bool)).fillna(False) == True) &
                          (passes["pass_length"].fillna(0) < 20)]) if "pass_shot_assist" in passes.columns else 0
    cc_vals = [len(crosses), len(long_passes), _short_a, len(through_balls)]
    cc_colors = [C_DARK, C_GREEN, C_YELLOW, C_ORANGE]
    bars4 = ax4.bar(cc_names, cc_vals, color=cc_colors, edgecolor="black", linewidth=0.7)
    total_cc = sum(cc_vals) if sum(cc_vals) > 0 else 1
    for bar, val in zip(bars4, cc_vals):
        pct = val / total_cc * 100
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cc_vals)*0.02,
                 f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax4.set_title("CHANCE CREATION MATRIX", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Count"); ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    # Panel 5: Passing style
    ax5 = fig_mega.add_subplot(gs[2, 0])
    _short = len(passes[passes["pass_length"].fillna(0) <= 15])
    _medium = len(passes[(passes["pass_length"].fillna(0) > 15) & (passes["pass_length"].fillna(0) <= 30)])
    _long_s = len(long_passes)
    style_names = ["Short\n(≤15)", "Medium\n(15-30)", "Long\n(>30)"]
    style_vals = [_short, _medium, _long_s]
    style_colors = [C_BLUE, C_GREEN, C_DARK]
    bars5 = ax5.bar(style_names, style_vals, color=style_colors, edgecolor="black", linewidth=0.6)
    total_s = sum(style_vals) if sum(style_vals) > 0 else 1
    for bar, val in zip(bars5, style_vals):
        pct = val / total_s * 100
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(style_vals)*0.01,
                 f"{val}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax5.set_title("PASSING STYLE BREAKDOWN", fontsize=13, fontweight="bold")
    ax5.set_ylabel("Count"); ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

    # Panel 6: Progression pie
    ax6 = fig_mega.add_subplot(gs[2, 1])
    _prog_c = len(progressive_passes); _nonprog_c = len(passes) - _prog_c
    ax6.pie([_prog_c, _nonprog_c], labels=["Progressive", "Non-Progressive"],
            autopct=lambda p: f"{p:.1f}%\n({int(round(p/100*(_prog_c+_nonprog_c)))})",
            colors=[C_GREEN, C_CORAL], startangle=140,
            textprops={"fontsize": 11}, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax6.set_title(f"PROGRESSION RATIO  |  Completion: {pass_completion:.1f}%", fontsize=13, fontweight="bold")

    fig_mega.suptitle(f"{selected_team} — TACTICAL BUILD-UP MEGA DASHBOARD\n"
                      "Bundesliga 2023/24  |  StatsBomb Event Data",
                      fontsize=17, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig_mega)
    plt.close(fig_mega)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: BUILD-UP STRUCTURE
# ═══════════════════════════════════════════════════════════════
elif section == "2. Build-up Structure":
    st.title(f"🏗️ {selected_team} — Build-up Structure")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sequences", f"{total_sequences:,}")
    c2.metric("Avg Length", f"{avg_seq_len:.1f}")
    c3.metric("Total Actions", f"{len(build):,}")
    c4.metric("Pass Completion", f"{pass_completion:.1f}%")

    st.markdown("---")

    # ZONE DISTRIBUTION
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Zone Distribution (Left / Center / Right)")
        if len(zone_counts) > 0:
            fig_zone = go.Figure(data=[go.Pie(
                labels=zone_counts.index.tolist(),
                values=zone_counts.values.tolist(),
                marker_colors=[C_DARK, C_GREEN, C_YELLOW],
                textinfo="label+percent+value",
                textfont_size=14,
                hole=0.3,
            )])
            fig_zone.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_zone, width="stretch")

    with col2:
        st.markdown("### Progressive vs Non-Progressive Passes")
        _prog_c = len(progressive_passes)
        _nonprog_c = len(passes) - _prog_c
        fig_prog = go.Figure(data=[go.Pie(
            labels=["Progressive", "Non-Progressive"],
            values=[_prog_c, _nonprog_c],
            marker_colors=[C_GREEN, C_CORAL],
            textinfo="label+percent+value",
            textfont_size=14,
            hole=0.3,
        )])
        fig_prog.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_prog, width="stretch")

    # HEATMAP
    st.markdown("### 🔥 Build-up Activity Heatmap")

    fig_heat, ax_h = plt.subplots(figsize=(14, 9))
    fig_heat.patch.set_facecolor("#0d2818")
    draw_pitch(ax_h)

    _bx = build["x"].dropna()
    _by = build["y"].dropna()
    ax_h.scatter(_bx, _by, alpha=0.06, s=6, color="#FFD700", zorder=2)
    sns.kdeplot(x=_bx, y=_by, fill=True, cmap="YlOrRd", alpha=0.5, levels=20, thresh=0.05, ax=ax_h, zorder=3)

    # Zone annotations
    _total = len(build)
    for zone_name, zone_x in [("LEFT", 30), ("CENTER", 60), ("RIGHT", 90)]:
        _cnt = len(build[build["zone"] == zone_name.capitalize()])
        ax_h.text(zone_x, 13, f"{zone_name}\n{_cnt} ({_cnt/_total*100:.1f}%)",
                  fontsize=11, color="cyan", ha="center", fontweight="bold", alpha=0.9)

    add_direction_arrows(ax_h)
    ax_h.set_title(f"{selected_team} — BUILD-UP ACTIVITY HEATMAP\nAll Build-up Sequences (≥3 actions)",
                   fontsize=15, fontweight="bold", color="white", pad=18)
    # Info bar
    ax_h.text(60, 84, f"Total Sequences: {total_sequences}  |  Total Actions: {_total}  |  Avg Length: {avg_seq_len:.1f}",
              fontsize=10, color="white", ha="center", fontweight="bold", style="italic", alpha=0.9)
    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close(fig_heat)

    dominant = zone_dist.idxmax() if len(zone_dist) > 0 else "N/A"
    insight_box(f"Build-up is **{dominant}-dominated** ({zone_dist.max():.1f}%). "
                f"The team constructs possessions of average **{avg_seq_len:.1f} actions**, "
                f"indicating a {'patient, multi-phase' if avg_seq_len > 20 else 'direct and efficient'} build-up style.")

    # Passing style breakdown
    st.markdown("### 🎯 Passing Style Breakdown")

    _short = len(passes[passes["pass_length"].fillna(0) <= 15])
    _medium = len(passes[(passes["pass_length"].fillna(0) > 15) & (passes["pass_length"].fillna(0) <= 30)])
    _long_s = len(long_passes)

    # Direction
    _fwd = len(passes[passes.apply(lambda r: (
        isinstance(r.get("pass_end_location"), list) and isinstance(r.get("location"), list) and
        len(r.get("pass_end_location", [])) > 0 and len(r.get("location", [])) > 0 and
        r["pass_end_location"][0] > r["location"][0]), axis=1)])
    _bwd = len(passes[passes.apply(lambda r: (
        isinstance(r.get("pass_end_location"), list) and isinstance(r.get("location"), list) and
        len(r.get("pass_end_location", [])) > 0 and len(r.get("location", [])) > 0 and
        r["pass_end_location"][0] < r["location"][0]), axis=1)])
    _lat = len(passes) - _fwd - _bwd

    style_labels = ["Short (≤15)", "Medium (15-30)", "Long (>30)", "Forward", "Backward", "Lateral"]
    style_values = [_short, _medium, _long_s, _fwd, _bwd, _lat]
    style_cs = [C_BLUE, C_GREEN, C_DARK, C_YELLOW, C_CORAL, C_ORANGE]

    fig_style = go.Figure()
    fig_style.add_trace(go.Bar(
        x=style_labels, y=style_values,
        marker_color=style_cs,
        text=[f"{v}<br>({v/sum(style_values)*100:.1f}%)" if sum(style_values) > 0 else str(v) for v in style_values],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig_style.update_layout(
        title="Pass Distance & Direction Profile",
        yaxis_title="Count", height=450, showlegend=False, plot_bgcolor="white",
    )
    st.plotly_chart(fig_style, width="stretch")


# ═══════════════════════════════════════════════════════════════
# SECTION 3: PROGRESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif section == "3. Progression Analysis":
    st.title(f"📈 {selected_team} — Progression Analysis")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Progressive Passes", f"{len(progressive_passes):,}")
    c2.metric("Line-Breaking", f"{len(line_breaking):,}")
    c3.metric("Progressive Carries", f"{len(progressive_carries):,}")
    c4.metric("Progression Rate", f"{prog_rate:.1f}%")
    c5.metric("Ball Prog. Index", f"{ball_prog_index:,}")

    st.markdown("---")

    # Progression actions bar
    st.markdown("### Progression Actions Breakdown")
    prog_metrics = {
        "Progressive Passes": len(progressive_passes),
        "Line-Breaking Passes": len(line_breaking),
        "Progressive Carries": len(progressive_carries),
        "Final Third Passes": len(final_third),
    }
    fig_prog_bar = go.Figure()
    fig_prog_bar.add_trace(go.Bar(
        x=list(prog_metrics.keys()), y=list(prog_metrics.values()),
        marker_color=[C_RED, C_BLUE, C_GREEN, C_YELLOW],
        text=list(prog_metrics.values()),
        textposition="outside",
        textfont=dict(size=14, color="black"),
    ))
    fig_prog_bar.update_layout(
        title="Build-up Progression Actions",
        yaxis_title="Count", height=450, showlegend=False, plot_bgcolor="white",
    )
    st.plotly_chart(fig_prog_bar, width="stretch")

    # Progression by pass distance (grouped bar)
    st.markdown("### Progression Profile by Pass Distance")

    _short_all = passes[passes["pass_length"].fillna(0) <= 15]
    _med_all = passes[(passes["pass_length"].fillna(0) > 15) & (passes["pass_length"].fillna(0) <= 30)]
    _long_all = passes[passes["pass_length"].fillna(0) > 30]

    def _count_prog(df):
        try:
            return int(df["progressive"].sum())
        except Exception:
            return 0

    cats = ["Short (≤15)", "Medium (15-30)", "Long (>30)"]
    p_vals = [_count_prog(_short_all), _count_prog(_med_all), _count_prog(_long_all)]
    np_vals = [len(_short_all) - p_vals[0], len(_med_all) - p_vals[1], len(_long_all) - p_vals[2]]

    fig_grouped = go.Figure()
    fig_grouped.add_trace(go.Bar(name="Progressive", x=cats, y=p_vals, marker_color=C_GREEN,
                                  text=p_vals, textposition="outside"))
    fig_grouped.add_trace(go.Bar(name="Non-Progressive", x=cats, y=np_vals, marker_color=C_CORAL,
                                  text=np_vals, textposition="outside"))
    fig_grouped.update_layout(
        barmode="group", title="Progressive vs Non-Progressive by Pass Distance",
        yaxis_title="Count", height=450, plot_bgcolor="white",
    )
    st.plotly_chart(fig_grouped, width="stretch")

    # Progressive pass map
    st.markdown("### 🗺️ Progressive Pass Map")

    fig_pp, ax_pp = plt.subplots(figsize=(14, 9))
    fig_pp.patch.set_facecolor("#0d2818")
    draw_pitch(ax_pp)

    _pp = progressive_passes.dropna(subset=["location", "pass_end_location"])
    if len(_pp) > 400:
        _pp_sample = _pp.sample(400, random_state=42)
    else:
        _pp_sample = _pp

    for _, row in _pp_sample.iterrows():
        loc = row["location"]
        end = row["pass_end_location"]
        if isinstance(loc, list) and isinstance(end, list) and len(loc) >= 2 and len(end) >= 2:
            ax_pp.annotate("",
                xy=(end[0], end[1]), xytext=(loc[0], loc[1]),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                color="#FF6B35", lw=0.8, alpha=0.5))

    _pp_sample_valid = _pp_sample.copy()
    _pp_sample_valid["sx"] = _pp_sample_valid["location"].apply(lambda l: l[0] if isinstance(l, list) and len(l) > 0 else np.nan)
    _pp_sample_valid["sy"] = _pp_sample_valid["location"].apply(lambda l: l[1] if isinstance(l, list) and len(l) > 1 else np.nan)
    ax_pp.scatter(_pp_sample_valid["sx"], _pp_sample_valid["sy"], s=12, color="#FFD700", alpha=0.6, zorder=5,
                  edgecolors="white", linewidth=0.3)

    add_direction_arrows(ax_pp)
    ax_pp.set_title(f"PROGRESSIVE PASS MAP — {len(progressive_passes)} Passes ({prog_rate:.1f}% of all)\n"
                    "Arrows = direction of progressive passes (≥10m forward gain)",
                    fontsize=14, fontweight="bold", color="white", pad=18)
    plt.tight_layout()
    st.pyplot(fig_pp)
    plt.close(fig_pp)

    insight_box(f"Progression rate of **{prog_rate:.1f}%** with **{ball_prog_index:,}** ball progression index "
                f"(passes + carries combined). "
                f"{'Direct and aggressive — the team advances quickly.' if prog_rate > 25 else 'Balanced between patience and directness.' if prog_rate > 15 else 'Patient progression — recycling possession before advancing.'}")


# ═══════════════════════════════════════════════════════════════
# SECTION 4: FINAL THIRD & CHANCE CREATION
# ═══════════════════════════════════════════════════════════════
elif section == "4. Final Third & Chance Creation":
    st.title(f"⚽ {selected_team} — Final Third & Chance Creation")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Final Third Passes", f"{len(final_third):,}")
    c2.metric("Zone 14 Entries", f"{len(z14_passes):,}")
    c3.metric("Crosses", f"{len(crosses):,}")
    c4.metric("Shot Assists", f"{len(shot_assists):,}")
    c5.metric("Through Balls", f"{len(through_balls):,}")

    st.markdown("---")

    # Chance Creation Matrix
    st.markdown("### Chance Creation Matrix")

    _short_a = len(passes[(passes.get("pass_shot_assist", pd.Series(dtype=bool)).fillna(False) == True) &
                          (passes["pass_length"].fillna(0) < 20)]) if "pass_shot_assist" in passes.columns else 0

    cc_labels = ["Crosses", "Long Passes", "Short Assists", "Through Balls"]
    cc_values = [len(crosses), len(long_passes), _short_a, len(through_balls)]
    cc_colors_list = [C_DARK, C_GREEN, C_YELLOW, C_ORANGE]

    total_cc = sum(cc_values) if sum(cc_values) > 0 else 1
    fig_cc = go.Figure()
    fig_cc.add_trace(go.Bar(
        x=cc_labels, y=cc_values,
        marker_color=cc_colors_list,
        text=[f"{v}<br>({v/total_cc*100:.1f}%)" for v in cc_values],
        textposition="outside",
        textfont=dict(size=13),
    ))
    fig_cc.update_layout(
        title="How Are Chances Being Generated?",
        yaxis_title="Count", xaxis_title="Creation Type",
        height=450, showlegend=False, plot_bgcolor="white",
    )
    st.plotly_chart(fig_cc, width="stretch")

    # Third-by-Third Pitch
    st.markdown("### 🏟️ Third-by-Third Zone Breakdown")

    col1, col2 = st.columns([1, 1])

    with col1:
        _def_c = len(build[build["x"] < 40])
        _mid_c = len(build[(build["x"] >= 40) & (build["x"] <= 80)])
        _att_c = len(build[build["x"] > 80])
        _total = len(build) if len(build) > 0 else 1

        third_df = pd.DataFrame({
            "Third": ["Defensive", "Middle", "Attacking"],
            "Actions": [_def_c, _mid_c, _att_c],
            "Percentage": [f"{_def_c/_total*100:.1f}%", f"{_mid_c/_total*100:.1f}%", f"{_att_c/_total*100:.1f}%"],
        })
        st.dataframe(third_df, width="stretch", hide_index=True)

        fig_third = go.Figure(data=[go.Pie(
            labels=["Defensive", "Middle", "Attacking"],
            values=[_def_c, _mid_c, _att_c],
            marker_colors=["cyan", C_YELLOW, C_RED],
            textinfo="label+percent+value",
            textfont_size=13, hole=0.35,
        )])
        fig_third.update_layout(height=350)
        st.plotly_chart(fig_third, width="stretch")

    with col2:
        fig_tz, ax_tz = plt.subplots(figsize=(10, 7))
        fig_tz.patch.set_facecolor("#0d2818")
        draw_pitch(ax_tz)

        ax_tz.axvspan(0, 40, alpha=0.15, color="cyan")
        ax_tz.axvspan(40, 80, alpha=0.15, color="yellow")
        ax_tz.axvspan(80, 120, alpha=0.15, color="red")

        ax_tz.plot([40, 40], [0, 80], color="white", linewidth=1.2, alpha=0.6, linestyle="--")
        ax_tz.plot([80, 80], [0, 80], color="white", linewidth=1.2, alpha=0.6, linestyle="--")

        for zone_label, zone_x, cnt in [("DEFENSIVE", 20, _def_c), ("MIDDLE", 60, _mid_c), ("ATTACKING", 100, _att_c)]:
            ax_tz.text(zone_x, 75, f"{zone_label}\n{cnt} ({cnt/_total*100:.1f}%)",
                       fontsize=10, color="white", ha="center", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6))

        _def_pts = build[build["x"] < 40]
        _mid_pts = build[(build["x"] >= 40) & (build["x"] <= 80)]
        _att_pts = build[build["x"] > 80]
        ax_tz.scatter(_def_pts["x"], _def_pts["y"], s=5, alpha=0.15, color="cyan", zorder=2)
        ax_tz.scatter(_mid_pts["x"], _mid_pts["y"], s=5, alpha=0.15, color="yellow", zorder=2)
        ax_tz.scatter(_att_pts["x"], _att_pts["y"], s=5, alpha=0.15, color="#FF6B35", zorder=2)

        add_direction_arrows(ax_tz)
        ax_tz.set_title("Pitch Zone Distribution", fontsize=13, fontweight="bold", color="white", pad=12)
        plt.tight_layout()
        st.pyplot(fig_tz)
        plt.close(fig_tz)

    ft_rate = len(final_third) / len(passes) * 100 if len(passes) > 0 else 0
    insight_box(f"Final third penetration rate: **{ft_rate:.1f}%** of build-up passes enter the attacking third. "
                f"**{len(z14_passes)}** passes penetrate Zone 14 (the danger zone directly in front of goal). "
                f"{'Aggressive penetration profile.' if ft_rate > 15 else 'Conservative entry — prioritises retention over risk.'}")


# ═══════════════════════════════════════════════════════════════
# SECTION 5: PLAYER IMPACT
# ═══════════════════════════════════════════════════════════════
elif section == "5. Player Impact":
    st.title(f"👤 {selected_team} — Player Impact Analysis")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top 10 Build-up Involvement")
        _top10 = build["player"].value_counts().head(10)
        fig_top = go.Figure()
        fig_top.add_trace(go.Bar(
            y=_top10.index[::-1].tolist(),
            x=_top10.values[::-1].tolist(),
            orientation="h",
            marker_color=C_BLUE,
            text=_top10.values[::-1].tolist(),
            textposition="outside",
            textfont=dict(size=12),
        ))
        fig_top.update_layout(
            title="Total Actions in Build-up Sequences",
            xaxis_title="Actions", height=500, showlegend=False, plot_bgcolor="white",
        )
        st.plotly_chart(fig_top, width="stretch")

    with col2:
        st.markdown("### Defensive Third Build-up (x < 40)")
        _def = build[build["x"] < 40]
        _def_top = _def["player"].value_counts().head(10)
        fig_def = go.Figure()
        fig_def.add_trace(go.Bar(
            y=_def_top.index[::-1].tolist(),
            x=_def_top.values[::-1].tolist(),
            orientation="h",
            marker_color=C_CORAL,
            text=_def_top.values[::-1].tolist(),
            textposition="outside",
            textfont=dict(size=12),
        ))
        fig_def.update_layout(
            title="Actions in Own Defensive Third",
            xaxis_title="Actions", height=500, showlegend=False, plot_bgcolor="white",
        )
        st.plotly_chart(fig_def, width="stretch")

    # Player Progression Contribution
    st.markdown("### 📊 Player Progression Contribution")

    prog_player = progressive_passes.groupby("player").size().sort_values(ascending=False).head(10)
    fig_pp_player = go.Figure()
    fig_pp_player.add_trace(go.Bar(
        y=prog_player.index[::-1].tolist(),
        x=prog_player.values[::-1].tolist(),
        orientation="h",
        marker_color=C_RED,
        text=prog_player.values[::-1].tolist(),
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig_pp_player.update_layout(
        title="Top 10 Players by Progressive Passes",
        xaxis_title="Progressive Passes", height=500, showlegend=False, plot_bgcolor="white",
    )
    st.plotly_chart(fig_pp_player, width="stretch")

    # Shot assist leaders
    if len(shot_assists) > 0:
        st.markdown("### 🎯 Shot Assist Leaders")
        sa_player = shot_assists.groupby("player").size().sort_values(ascending=False).head(10)
        fig_sa = go.Figure()
        fig_sa.add_trace(go.Bar(
            y=sa_player.index[::-1].tolist(),
            x=sa_player.values[::-1].tolist(),
            orientation="h",
            marker_color=C_GREEN,
            text=sa_player.values[::-1].tolist(),
            textposition="outside",
        ))
        fig_sa.update_layout(
            title="Top Shot Assist Providers",
            xaxis_title="Shot Assists", height=400, showlegend=False, plot_bgcolor="white",
        )
        st.plotly_chart(fig_sa, width="stretch")

    top3 = build["player"].value_counts().head(3)
    insight_box(f"Top 3 build-up contributors: **{', '.join(top3.index.tolist())}**. "
                f"The top player ({top3.index[0]}) accounts for {top3.iloc[0]:,} actions "
                f"({top3.iloc[0]/len(build)*100:.1f}% of all build-up activity).")


# ═══════════════════════════════════════════════════════════════
# SECTION 6: BENCHMARK COMPARISON
# ═══════════════════════════════════════════════════════════════
elif section == "6. Benchmark Comparison":
    st.title(f"📊 {selected_team} — Benchmark Comparison")
    st.markdown("Comparing against estimated Bundesliga averages (computed from all teams in the dataset).")

    st.markdown("---")

    # Compute league averages from all teams
    @st.cache_data(show_spinner="Computing league benchmarks…")
    def compute_league_benchmarks(_events_all, _matches_all, comp_team):
        """Compute average metrics across all teams for benchmarking."""
        all_team_names = sorted(pd.unique(_matches_all[["home_team", "away_team"]].values.ravel()))
        league_data = []

        for t_name in all_team_names:
            t_ev = _events_all[_events_all["team"] == t_name].copy()
            if len(t_ev) == 0:
                continue
            t_ev["possession_id"] = t_ev["match_id"].astype(str) + "_" + t_ev["possession"].astype(str)
            b_types = ["Pass", "Carry", "Ball Receipt*"]
            t_build = t_ev[t_ev["type"].isin(b_types)].copy()
            t_build = t_build.groupby("possession_id").filter(lambda x: len(x) >= 3)

            if len(t_build) == 0:
                continue

            t_build["x"] = t_build["location"].apply(safe_x)
            t_passes = t_build[t_build["type"] == "Pass"].copy()
            t_passes["outcome"] = t_passes["pass_outcome"].fillna("Complete")
            t_passes["progressive"] = t_passes.apply(is_progressive, axis=1)

            _seqs = t_build["possession_id"].nunique()
            _avg = t_build.groupby("possession_id").size().mean() if _seqs > 0 else 0
            _pc = (t_passes["outcome"] == "Complete").mean() * 100 if len(t_passes) > 0 else 0
            _pr = t_passes["progressive"].mean() * 100 if len(t_passes) > 0 else 0
            _ft = len(t_passes[t_passes["x"] > 80])

            league_data.append({
                "team": t_name,
                "sequences": _seqs,
                "avg_length": _avg,
                "pass_completion": _pc,
                "prog_rate": _pr,
                "final_third": _ft,
                "total_passes": len(t_passes),
            })

        return pd.DataFrame(league_data)

    # We need ALL events (not just selected team) for benchmarks
    # Load all events across the league
    @st.cache_data(show_spinner="Loading full league data for benchmarks…")
    def load_all_league_events(comp_id, season_id):
        from statsbombpy import sb
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        all_ev = []
        for mid in matches["match_id"].unique():
            ev = sb.events(match_id=mid)
            ev["match_id"] = mid
            all_ev.append(ev)
        return pd.concat(all_ev, ignore_index=True), matches

    try:
        all_league_events, all_league_matches = load_all_league_events(COMP_ID, SEASON_ID)
        league_df = compute_league_benchmarks(all_league_events, all_league_matches, selected_team)

        if len(league_df) > 0:
            league_avg = league_df.drop(columns=["team"]).mean()
            team_row = league_df[league_df["team"] == selected_team]

            # Comparison table
            metrics_compare = ["pass_completion", "prog_rate", "avg_length", "sequences", "final_third"]
            metric_labels = ["Pass Completion %", "Progression Rate %", "Avg Sequence Length", "Build-up Sequences", "Final Third Passes"]

            team_vals = []
            avg_vals = []
            for m in metrics_compare:
                team_vals.append(team_row[m].values[0] if len(team_row) > 0 else 0)
                avg_vals.append(league_avg[m])

            # Radar chart
            st.markdown("### Radar: Team vs League Average")

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[tv / av * 100 if av > 0 else 0 for tv, av in zip(team_vals, avg_vals)],
                theta=metric_labels,
                fill="toself",
                name=selected_team,
                line_color=C_RED,
                fillcolor="rgba(230,57,70,0.3)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[100] * len(metric_labels),
                theta=metric_labels,
                fill="toself",
                name="League Average (100%)",
                line_color=C_BLUE,
                fillcolor="rgba(69,123,157,0.15)",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 200])),
                showlegend=True, height=500,
                title="Team Performance Relative to League Average (100% = avg)",
            )
            st.plotly_chart(fig_radar, width="stretch")

            # Bar comparison
            st.markdown("### Side-by-Side Comparison")

            fig_compare = make_subplots(rows=1, cols=len(metrics_compare),
                                         subplot_titles=metric_labels)
            for i, (m, label) in enumerate(zip(metrics_compare, metric_labels)):
                tv = team_vals[i]
                av = avg_vals[i]
                fig_compare.add_trace(go.Bar(
                    x=[selected_team, "League Avg"],
                    y=[tv, av],
                    marker_color=[C_RED, C_BLUE],
                    text=[f"{tv:.1f}", f"{av:.1f}"],
                    textposition="outside",
                    showlegend=False,
                ), row=1, col=i+1)
            fig_compare.update_layout(height=400, plot_bgcolor="white")
            st.plotly_chart(fig_compare, width="stretch")

            # League ranking table
            st.markdown("### 🏆 League Rankings")
            for m, label in zip(metrics_compare, metric_labels):
                ranked = league_df.sort_values(m, ascending=False).reset_index(drop=True)
                rank = ranked[ranked["team"] == selected_team].index[0] + 1 if len(ranked[ranked["team"] == selected_team]) > 0 else "N/A"
                val = team_row[m].values[0] if len(team_row) > 0 else 0
                st.write(f"**{label}**: {val:.1f} — Rank **#{rank}** of {len(ranked)} teams")

            insight_box(f"Benchmark analysis across {len(league_df)} Bundesliga teams. "
                        f"Check pass completion and progression rate relative to the league median to identify competitive advantages.")
        else:
            st.warning("Could not compute league benchmarks. Insufficient data.")

    except Exception as e:
        st.warning(f"League benchmark data requires loading all matches. This may take time. Error: {e}")
        st.info("Showing simplified comparison with estimated averages instead.")

        # Fallback: estimated Bundesliga averages
        est_avg = {"Pass Completion %": 80.0, "Progression Rate %": 20.0, "Final Third Passes": 4000}
        team_actual = {"Pass Completion %": pass_completion, "Progression Rate %": prog_rate, "Final Third Passes": len(final_third)}

        fig_est = go.Figure()
        fig_est.add_trace(go.Bar(name=selected_team, x=list(est_avg.keys()), y=list(team_actual.values()), marker_color=C_RED))
        fig_est.add_trace(go.Bar(name="Est. League Avg", x=list(est_avg.keys()), y=list(est_avg.values()), marker_color=C_BLUE))
        fig_est.update_layout(barmode="group", height=450, plot_bgcolor="white")
        st.plotly_chart(fig_est, width="stretch")


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MATCH PREPARATION
# ═══════════════════════════════════════════════════════════════
elif section == "7. Match Preparation":
    st.title(f"📋 {selected_team} — Match Preparation Report")

    st.markdown("---")

    # ── Gather per-match data ──
    match_ids = sorted(build["match_id"].unique())
    match_labels_map = {}
    for _, row in team_matches.iterrows():
        mid = row["match_id"]
        opp = row["away_team"] if row["home_team"] == selected_team else row["home_team"]
        venue = "H" if row["home_team"] == selected_team else "A"
        date = str(row.get("match_date", ""))[:10]
        match_labels_map[mid] = {"full": f"vs {opp} ({date})", "opp": opp, "venue": venue, "date": date}

    match_data = []
    for mid in match_ids:
        info = match_labels_map.get(mid, {"full": str(mid), "opp": str(mid), "venue": "?", "date": ""})
        m_build = build[build["match_id"] == mid]
        m_pass = passes[passes["match_id"] == mid]

        prog_val = int(m_pass["progressive"].sum()) if "progressive" in m_pass.columns else 0
        ft_val = len(m_pass[m_pass["x"] > 80])
        cross_val = len(m_pass[m_pass["pass_cross"].fillna(False) == True]) if "pass_cross" in m_pass.columns else 0

        match_data.append({
            "mid": mid,
            "opp": info["opp"],
            "venue": info["venue"],
            "date": info["date"],
            "label": f"{info['opp'][:12]} ({info['date'][5:]})",
            "seqs": m_build["possession_id"].nunique(),
            "total_pass": len(m_pass),
            "prog": prog_val,
            "ft": ft_val,
            "cross": cross_val,
        })

    home_data = [d for d in match_data if d["venue"] == "H"]
    away_data = [d for d in match_data if d["venue"] == "A"]

    # ── Helper to build bar + volume charts for a subset ──
    def render_match_charts(data, venue_label, venue_emoji):
        if not data:
            st.info(f"No {venue_label.lower()} matches in the current selection.")
            return

        labels = [d["label"] for d in data]
        prog = [d["prog"] for d in data]
        ft = [d["ft"] for d in data]
        cross = [d["cross"] for d in data]
        seqs = [d["seqs"] for d in data]
        total_pass = [d["total_pass"] for d in data]

        # --- Offensive metrics grouped bar ---
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Progressive Passes", x=labels, y=prog, marker_color=C_GREEN,
            text=prog, textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Progressive: %{y}<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            name="Final Third Passes", x=labels, y=ft, marker_color=C_YELLOW,
            text=ft, textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Final Third: %{y}<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            name="Crosses", x=labels, y=cross, marker_color=C_CORAL,
            text=cross, textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Crosses: %{y}<extra></extra>",
        ))
        fig_bar.update_layout(
            barmode="group",
            title=f"{venue_emoji} {venue_label} — Offensive Metrics per Match",
            yaxis_title="Count", height=480, plot_bgcolor="white",
            xaxis_tickangle=-45, xaxis_tickfont=dict(size=9),
            margin=dict(t=60, b=100),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            bargap=0.2, bargroupgap=0.08,
        )
        st.plotly_chart(fig_bar, width="stretch")

        # --- Volume: sequences bar + total passes line ---
        fig_vol = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=(f"Build-up Sequences ({venue_label})", f"Total Passes ({venue_label})"),
            row_heights=[0.5, 0.5],
        )
        fig_vol.add_trace(go.Bar(
            name="Sequences", x=labels, y=seqs, marker_color=C_BLUE,
            text=seqs, textposition="outside", textfont=dict(size=9),
            hovertemplate="<b>%{x}</b><br>Sequences: %{y}<extra></extra>",
        ), row=1, col=1)
        fig_vol.add_trace(go.Scatter(
            name="Total Passes", x=labels, y=total_pass,
            mode="lines+markers+text", text=total_pass, textposition="top center",
            textfont=dict(size=9, color=C_RED),
            line=dict(color=C_RED, width=2.5), marker=dict(size=7, color=C_RED),
            hovertemplate="<b>%{x}</b><br>Passes: %{y}<extra></extra>",
        ), row=2, col=1)
        fig_vol.update_layout(
            height=580, plot_bgcolor="white",
            margin=dict(t=50, b=80),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        )
        fig_vol.update_xaxes(tickangle=-45, tickfont=dict(size=9), row=2, col=1)
        fig_vol.update_yaxes(title_text="Sequences", row=1, col=1)
        fig_vol.update_yaxes(title_text="Total Passes", row=2, col=1)
        st.plotly_chart(fig_vol, width="stretch")

        # Quick summary metrics for this venue
        avg_prog = np.mean(prog)
        avg_ft = np.mean(ft)
        avg_seqs = np.mean(seqs)
        avg_passes = np.mean(total_pass)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Avg Prog. Passes ({venue_label[0]})", f"{avg_prog:.0f}")
        c2.metric(f"Avg Final Third ({venue_label[0]})", f"{avg_ft:.0f}")
        c3.metric(f"Avg Sequences ({venue_label[0]})", f"{avg_seqs:.0f}")
        c4.metric(f"Avg Passes ({venue_label[0]})", f"{avg_passes:.0f}")

    # ── RENDER HOME SECTION ──
    st.markdown("### 🏠 Home Matches")
    st.caption(f"{len(home_data)} home matches")
    render_match_charts(home_data, "Home", "🏠")

    st.markdown("---")

    # ── RENDER AWAY SECTION ──
    st.markdown("### ✈️ Away Matches")
    st.caption(f"{len(away_data)} away matches")
    render_match_charts(away_data, "Away", "✈️")

    # Coach-readable summary
    st.markdown("---")
    st.markdown("### 📝 Coach-Readable Tactical Summary")

    dominant = zone_dist.idxmax() if len(zone_dist) > 0 else "N/A"
    top3 = build["player"].value_counts().head(3)
    _def_top3 = build[build["x"] < 40]["player"].value_counts().head(3)
    ft_rate = len(final_third) / len(passes) * 100 if len(passes) > 0 else 0

    report = f"""
---

**🏟️ {selected_team.upper()} — TACTICAL BUILD-UP INTELLIGENCE REPORT**
**Bundesliga 2023/24** | **{len(selected_matches)} Matches Analysed** | StatsBomb Event Data

---

**1. BUILD-UP STRUCTURE**
- Total Build-up Sequences: **{total_sequences:,}**
- Average Sequence Length: **{avg_seq_len:.2f} actions**
- Total Build-up Actions: **{len(build):,}**

**2. ZONE DOMINANCE**
"""
    for z in zone_dist.index:
        bar = "█" * int(zone_dist[z] / 2)
        report += f"- {z}: **{zone_dist[z]:.1f}%** {bar}\n"
    report += f"- Dominant channel: **{dominant}** ({zone_dist.max():.1f}%)\n"

    report += f"""
**3. PROGRESSION STYLE**
- Progressive Passes: **{len(progressive_passes):,}**
- Line-Breaking Passes: **{len(line_breaking):,}**
- Pass Completion Rate: **{pass_completion:.1f}%**
- Progression Rate: **{prog_rate:.1f}%**
- Ball Progression Index: **{ball_prog_index:,}**

**4. FINAL THIRD / ZONE 14**
- Final Third Passes: **{len(final_third):,}**
- Zone 14 Entries: **{len(z14_passes):,}**
- Final Third Penetration Rate: **{ft_rate:.1f}%**

**5. CHANCE CREATION**
- Crosses: **{len(crosses):,}**
- Long Passes: **{len(long_passes):,}**
- Shot Assists: **{len(shot_assists):,}**
- Through Balls: **{len(through_balls):,}**

**6. KEY PLAYERS**
- Top 3 Build-up: **{', '.join([f'{p} ({c})' for p, c in top3.items()])}**
- Top 3 Defensive Third: **{', '.join([f'{p} ({c})' for p, c in _def_top3.items()])}**

**7. TACTICAL INTERPRETATION**
"""
    # Structure
    if dominant == "Center":
        report += "- 🏗️ Central build-up preference — possession-based system through midfield.\n"
    elif dominant == "Left":
        report += "- 🏗️ Left-sided build-up bias — overloading through the left channel.\n"
    else:
        report += "- 🏗️ Right-sided emphasis — primary progression through the right corridor.\n"

    if prog_rate > 25:
        report += "- 📈 High progression rate — vertical, direct ball advancement.\n"
    elif prog_rate > 15:
        report += "- 📈 Moderate progression — balanced patience and directness.\n"
    else:
        report += "- 📈 Low progression — patient possession-recycling style.\n"

    if ft_rate > 15:
        report += "- ⚽ Aggressive final-third penetration.\n"
    else:
        report += "- ⚽ Conservative final-third entry — retention over risk.\n"

    report += f"- 🎯 {'Elite-level' if pass_completion > 85 else 'Strong'} pass accuracy at {pass_completion:.1f}%.\n"
    report += "\n---\n*Report auto-generated from StatsBomb open data.*"

    st.markdown(report)

    # Export
    st.markdown("---")
    st.markdown("### 📥 Export Report")
    st.download_button(
        "⬇️ Download Report as Text",
        data=report,
        file_name=f"{selected_team.replace(' ', '_')}_buildup_report.md",
        mime="text/markdown",
    )

    # Export data
    export_df = pd.DataFrame({
        "Metric": [
            "Build-up Sequences", "Avg Sequence Length", "Pass Completion %",
            "Progressive Passes", "Progression Rate %", "Ball Progression Index",
            "Final Third Passes", "Zone 14 Entries",
            "Crosses", "Long Passes", "Shot Assists", "Through Balls",
        ],
        "Value": [
            total_sequences, round(avg_seq_len, 2), round(pass_completion, 2),
            len(progressive_passes), round(prog_rate, 2), ball_prog_index,
            len(final_third), len(z14_passes),
            len(crosses), len(long_passes), len(shot_assists), len(through_balls),
        ],
    })

    csv = export_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download KPI Data as CSV",
        data=csv,
        file_name=f"{selected_team.replace(' ', '_')}_kpis.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("Built with **Streamlit** + **StatsBomb** Open Data")
st.sidebar.markdown("*Bayer Leverkusen Tactical Dashboard*")
