"""
app.py — AI Debate Engine Streamlit UI
Live streaming debate with real-time scoring, round cards,
leaderboard, and final verdict animation.
"""

import sys 
import json
import time
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent / "src"))
from debate_engine import (
    DEBATER_PERSONAS, DEBATE_FORMATS,
    run_debate, DebateTurn, JudgeScore, DebateResult,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Debate Engine",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #07080d; }

  .hero {
    background: linear-gradient(135deg, #1a0520 0%, #07080d 45%, #051520 100%);
    border: 1px solid #2a1a3a;
    border-radius: 16px;
    padding: 36px 40px;
    text-align: center;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "VS";
    position: absolute;
    font-size: 120px;
    font-weight: 900;
    color: rgba(255,255,255,0.03);
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-family: 'Playfair Display', serif;
  }
  .hero h1 { font-size: 40px; font-weight: 700; color: #fff; margin: 0 0 6px; position:relative; }
  .hero p  { color: #64748b; font-size: 14px; margin: 0; position:relative; }

  .debater-card {
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
  }
  .card-pro { background: #0a0310; border: 1px solid #7f1d1d; border-left: 4px solid #ef4444; }
  .card-con { background: #030a1a; border: 1px solid #1e3a5f; border-left: 4px solid #3b82f6; }

  .argument-bubble-pro {
    background: linear-gradient(135deg, #1a0308, #0f020a);
    border: 1px solid #7f1d1d;
    border-radius: 16px 16px 4px 16px;
    padding: 16px 20px;
    margin: 10px 0 10px 60px;
    font-size: 14px;
    line-height: 1.8;
    color: #fca5a5;
  }
  .argument-bubble-con {
    background: linear-gradient(135deg, #030d1a, #020810);
    border: 1px solid #1e3a5f;
    border-radius: 16px 16px 16px 4px;
    padding: 16px 20px;
    margin: 10px 60px 10px 0;
    font-size: 14px;
    line-height: 1.8;
    color: #93c5fd;
  }
  .speaker-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .round-header {
    text-align: center;
    padding: 8px;
    margin: 16px 0 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #475569;
    border-top: 1px solid #1a2030;
    border-bottom: 1px solid #1a2030;
  }
  .score-row {
    background: #080a12;
    border: 1px solid #1a2030;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
  }
  .score-bar-fill { height: 6px; border-radius: 3px; transition: width 0.3s; }

  .verdict-box {
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin: 20px 0;
  }
  .verdict-pro { background: #0f0205; border: 2px solid #ef4444; }
  .verdict-con { background: #020810; border: 2px solid #3b82f6; }
  .verdict-tie { background: #0a0a05; border: 2px solid #f59e0b; }

  .stat-card {
    background: #080a12;
    border: 1px solid #1a2030;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }
  .stat-val   { font-size: 22px; font-weight: 700; }
  .stat-label { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 3px; }

  div.stButton > button {
    background: linear-gradient(135deg, #7c2d92, #a855f7);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 16px;
    width: 100%;
    letter-spacing: 0.5px;
  }
  div.stButton > button:hover { opacity: 0.85; }

  .topic-examples {
    background: #080a12;
    border: 1px solid #1a2030;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #64748b;
    margin: 3px 0;
    cursor: pointer;
  }
</style>
""", unsafe_allow_html=True)

# ── Sample topics ──────────────────────────────────────────────────────────────
SAMPLE_TOPICS = [
    "AI will replace more jobs than it creates",
    "Social media does more harm than good to society",
    "Remote work is better than office work",
    "Cryptocurrency is the future of finance",
    "Space exploration should be humanity's top priority",
    "Universal Basic Income should be implemented globally",
    "Nuclear energy is essential for solving climate change",
    "Standardised testing should be abolished in education",
    "The 4-day work week should become the global standard",
    "Gene editing in humans should be allowed for disease prevention",
    "AGI will be developed within 10 years",
    "Open source AI is safer than closed source AI",
]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎤 Debate Settings")
    st.markdown("---")

    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("Free Gemini API Key", type="password", placeholder="AIza...")
    if not api_key:
        st.info("🆓 Free at [aistudio.google.com](https://aistudio.google.com)")

    st.markdown("---")
    st.markdown("### 🥊 Debater Personas")
    persona_a_key = st.selectbox(
        "PRO Debater 🔴",
        list(DEBATER_PERSONAS.keys()),
        format_func=lambda x: f"{DEBATER_PERSONAS[x]['name']} — {DEBATER_PERSONAS[x]['style'][:30]}...",
        index=0,
    )
    persona_b_key = st.selectbox(
        "CON Debater 🔵",
        list(DEBATER_PERSONAS.keys()),
        format_func=lambda x: f"{DEBATER_PERSONAS[x]['name']} — {DEBATER_PERSONAS[x]['style'][:30]}...",
        index=1,
    )
    if persona_a_key == persona_b_key:
        st.warning("⚠️ Both debaters have the same persona — choose different ones for a better debate!")

    st.markdown("---")
    st.markdown("### 📋 Debate Format")
    format_key = st.selectbox(
        "Format",
        list(DEBATE_FORMATS.keys()),
        format_func=lambda x: f"{DEBATE_FORMATS[x]['name']} ({len(DEBATE_FORMATS[x]['rounds'])} rounds)",
    )
    fmt_info = DEBATE_FORMATS[format_key]
    st.caption(fmt_info["description"])

    st.markdown("---")
    st.markdown("### 🎲 Random Topic")
    if st.button("🎲 Surprise me!"):
        import random
        st.session_state["topic_input"] = random.choice(SAMPLE_TOPICS)

# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎤 Real-time AI Debate Engine</h1>
  <p>Two AI debaters with distinct personas argue any topic — judged live by a neutral AI</p>
</div>
""", unsafe_allow_html=True)

# ── Debater preview cards ──────────────────────────────────────────────────────
pa = DEBATER_PERSONAS[persona_a_key]
pb = DEBATER_PERSONAS[persona_b_key]

col_a, col_vs, col_b = st.columns([5, 1, 5])
with col_a:
    st.markdown(f"""
<div class="debater-card card-pro">
  <div style="font-size:24px;font-weight:700;color:#ef4444;">{pa['color']} {pa['name']}</div>
  <div style="font-size:11px;color:#7f1d1d;letter-spacing:2px;text-transform:uppercase;margin:4px 0;">PRO POSITION</div>
  <div style="font-size:12px;color:#fca5a5;line-height:1.5;">{pa['style']}</div>
</div>""", unsafe_allow_html=True)

with col_vs:
    st.markdown("""
<div style="text-align:center;padding:24px 0;font-size:28px;font-weight:900;color:#334155;">VS</div>
""", unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
<div class="debater-card card-con">
  <div style="font-size:24px;font-weight:700;color:#3b82f6;">{pb['color']} {pb['name']}</div>
  <div style="font-size:11px;color:#1e3a5f;letter-spacing:2px;text-transform:uppercase;margin:4px 0;">CON POSITION</div>
  <div style="font-size:12px;color:#93c5fd;line-height:1.5;">{pb['style']}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Topic input ────────────────────────────────────────────────────────────────
topic = st.text_input(
    "💬 Debate Topic",
    value=st.session_state.get("topic_input", ""),
    placeholder="e.g. AI will replace more jobs than it creates",
    label_visibility="visible",
)

st.markdown("**💡 Try one of these:**")
ex_cols = st.columns(4)
for i, example in enumerate(SAMPLE_TOPICS[:8]):
    with ex_cols[i % 4]:
        if st.button(f"💬 {example[:35]}...", key=f"ex_{i}"):
            st.session_state["topic_input"] = example
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
start_debate = st.button(f"🎤 START DEBATE — {fmt_info['name']}")

# ── Live debate ────────────────────────────────────────────────────────────────
if start_debate:
    if not topic.strip():
        st.warning("⚠️ Please enter a debate topic.")
    elif not api_key:
        st.error("⚠️ Please add your free Gemini API key in the sidebar.")
    elif persona_a_key == persona_b_key:
        st.error("⚠️ Please choose different personas for each debater.")
    else:
        # ── Live debate state ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### 🎤 Debate: *\"{topic}\"*")

        debate_container  = st.container()
        score_placeholder = st.empty()
        status_placeholder = st.empty()

        all_turns:  list[DebateTurn]  = []
        all_scores: list[JudgeScore]  = []

        # Placeholders for live rendering
        turn_placeholders = {}

        def on_turn(turn: DebateTurn):
            all_turns.append(turn)
            with debate_container:
                is_pro = turn.position == "PRO"

                # Round header
                if len(all_turns) == 1 or all_turns[-2].round_num != turn.round_num:
                    st.markdown(f'<div class="round-header">◈ {turn.round_type.upper()} — Round {turn.round_num}</div>', unsafe_allow_html=True)

                persona = DEBATER_PERSONAS[turn.persona]
                if is_pro:
                    st.markdown(f'<div class="speaker-label" style="color:#ef4444;margin-left:4px;">{persona["color"]} {turn.speaker} — PRO</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="argument-bubble-pro">{turn.argument}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="speaker-label" style="color:#3b82f6;text-align:right;margin-right:4px;">{persona["color"]} {turn.speaker} — CON</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="argument-bubble-con">{turn.argument}</div>', unsafe_allow_html=True)

            status_placeholder.markdown(f"⏳ *{turn.speaker} argued — Judge scoring...*")

        def on_score(score: JudgeScore):
            all_scores.append(score)
            # Update live scoreboard
            pro_scores  = [s for s in all_scores if any(t.speaker == s.speaker and t.position == "PRO" for t in all_turns)]
            con_scores  = [s for s in all_scores if any(t.speaker == s.speaker and t.position == "CON" for t in all_turns)]
            avg_pro = sum(s.total for s in pro_scores) / max(len(pro_scores), 1)
            avg_con = sum(s.total for s in con_scores) / max(len(con_scores), 1)

            with score_placeholder.container():
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#ef4444;">{avg_pro:.1f}</div><div class="stat-label">{pa["name"]} avg score</div></div>', unsafe_allow_html=True)
                with sc2:
                    lead = pa["name"] if avg_pro > avg_con else (pb["name"] if avg_con > avg_pro else "TIE")
                    lead_color = "#ef4444" if avg_pro > avg_con else ("#3b82f6" if avg_con > avg_pro else "#f59e0b")
                    st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:{lead_color};">{lead}</div><div class="stat-label">Leading</div></div>', unsafe_allow_html=True)
                with sc3:
                    st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#3b82f6;">{avg_con:.1f}</div><div class="stat-label">{pb["name"]} avg score</div></div>', unsafe_allow_html=True)

        # ── Run the debate ─────────────────────────────────────────────────────
        try:
            with st.spinner("🤖 Initialising debate..."):
                pass

            status_placeholder.markdown("🎤 *Debate starting...*")

            result = run_debate(
                api_key=api_key,
                topic=topic,
                persona_a_key=persona_a_key,
                persona_b_key=persona_b_key,
                format_key=format_key,
                on_turn=on_turn,
                on_score=on_score,
            )

            status_placeholder.empty()

            # ── Final verdict ──────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 🏆 Final Verdict")

            winner_name = pa["name"] if result.winner == "PRO" else (pb["name"] if result.winner == "CON" else "TIE")
            if result.winner == "PRO":
                verdict_class = "verdict-pro"
                winner_color  = "#ef4444"
                winner_emoji  = "🔴"
            elif result.winner == "CON":
                verdict_class = "verdict-con"
                winner_color  = "#3b82f6"
                winner_emoji  = "🔵"
            else:
                verdict_class = "verdict-tie"
                winner_color  = "#f59e0b"
                winner_emoji  = "🟡"

            st.markdown(f"""
<div class="verdict-box {verdict_class}">
  <div style="font-size:11px;letter-spacing:3px;color:#475569;text-transform:uppercase;margin-bottom:8px;">🏆 THE WINNER IS</div>
  <div style="font-size:44px;font-weight:700;color:{winner_color};">{winner_emoji} {winner_name}</div>
  <div style="font-size:14px;color:#94a3b8;margin-top:12px;line-height:1.7;">{result.final_verdict}</div>
</div>
""", unsafe_allow_html=True)

            # ── Score breakdown ────────────────────────────────────────────────
            st.markdown("### 📊 Score Analysis")

            pro_s = [s for s in result.scores if any(t.speaker == s.speaker and t.position == "PRO" for t in result.turns)]
            con_s = [s for s in result.scores if any(t.speaker == s.speaker and t.position == "CON" for t in result.turns)]

            def avg_cat(scores, cat):
                vals = [getattr(s, cat) for s in scores]
                return sum(vals) / max(len(vals), 1)

            categories = ["logic", "evidence", "rhetoric", "rebuttal"]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=f"{pa['name']} (PRO)",
                x=categories,
                y=[avg_cat(pro_s, c) for c in categories],
                marker_color="#ef4444",
                opacity=0.85,
            ))
            fig.add_trace(go.Bar(
                name=f"{pb['name']} (CON)",
                x=categories,
                y=[avg_cat(con_s, c) for c in categories],
                marker_color="#3b82f6",
                opacity=0.85,
            ))
            fig.update_layout(
                paper_bgcolor="#07080d", plot_bgcolor="#07080d",
                font_color="#94a3b8",
                yaxis=dict(title="Average Score (1-10)", range=[0, 10], gridcolor="#1a2030"),
                xaxis=dict(gridcolor="#1a2030"),
                barmode="group",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                height=320, margin=dict(t=20,b=20,l=10,r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Summary ────────────────────────────────────────────────────────
            st.markdown("### 📝 Debate Summary")
            st.markdown(f'<div style="background:#080a12;border:1px solid #1a2030;border-radius:10px;padding:18px 22px;font-size:14px;color:#94a3b8;line-height:1.8;">{result.final_summary}</div>', unsafe_allow_html=True)

            # Stats
            s1,s2,s3,s4 = st.columns(4)
            rounds_won_pro = sum(1 for s in result.scores if s.winner_round == "PRO")
            rounds_won_con = sum(1 for s in result.scores if s.winner_round == "CON")
            total_words    = sum(t.word_count for t in result.turns)

            for col, (val, label) in zip([s1,s2,s3,s4], [
                (result.total_rounds,        "Total Rounds"),
                (rounds_won_pro,             f"{pa['name']} Rounds Won"),
                (rounds_won_con,             f"{pb['name']} Rounds Won"),
                (f"{result.duration_sec:.0f}s", "Duration"),
            ]):
                with col:
                    st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#a78bfa;">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

            # Download
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Full Debate Transcript (.json)",
                data=json.dumps(asdict(result), indent=2, default=str),
                file_name=f"debate_{topic[:30].replace(' ','_')}.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"Debate failed: {str(e)}\n\nCheck your Gemini API key and try again.")

else:
    # Empty state
    st.markdown("""
<div style="text-align:center;padding:40px 20px;">
  <div style="font-size:64px;margin-bottom:16px;">🎤</div>
  <h3 style="color:#475569;">Enter a topic and click Start Debate</h3>
  <p style="color:#334155;font-size:14px;max-width:520px;margin:0 auto;">
    Two AI debaters with distinct personalities will argue opposing sides of any topic,
    scored live by a neutral AI judge across logic, evidence, rhetoric, and rebuttal.
  </p>
</div>
""", unsafe_allow_html=True)
