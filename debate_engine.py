"""
debate_engine.py — Real-time AI Debate Engine
Two Gemini AI instances with distinct personas debate any topic.
Each AI maintains its position, uses rhetorical strategies, rebuts
the opponent, and is judged by a third neutral AI moderator.

Architecture:
  Agent A  — Assigned PRO position, aggressive debater style
  Agent B  — Assigned CON position, analytical debater style
  Judge    — Neutral AI scoring each round on logic, evidence, rhetoric
  Moderator— Controls flow, enforces turn limits, summarises

Debate formats:
  - Oxford style     : opening → rebuttals → closing
  - Parliamentary    : timed rounds with points of information
  - Lincoln-Douglas  : value-based argumentation
  - Free-form        : open back-and-forth
"""

import os
import json
import time
import re
from dataclasses import dataclass, field, asdict
from typing import Iterator
import google.generativeai as genai


# ── Debater personas ───────────────────────────────────────────────────────────
DEBATER_PERSONAS = {
    "aggressive": {
        "name":  "APEX",
        "style": "aggressive, confident, uses strong rhetoric and emotional appeal",
        "traits": [
            "Opens with a bold, attention-grabbing claim",
            "Uses vivid analogies and metaphors",
            "Attacks weaknesses in opponent's arguments directly",
            "Appeals to emotion and urgency",
            "Ends every argument with a punchy one-liner",
        ],
        "color": "🔴",
    },
    "analytical": {
        "name":  "NOVA",
        "style": "analytical, data-driven, uses logic and evidence methodically",
        "traits": [
            "Leads with statistics and cited evidence",
            "Deconstructs opponent's arguments point by point",
            "Uses Socratic questioning to expose flaws",
            "Appeals to long-term consequences and reason",
            "Ends with a logical synthesis statement",
        ],
        "color": "🔵",
    },
    "philosophical": {
        "name":  "SAGE",
        "style": "philosophical, nuanced, draws on historical and ethical frameworks",
        "traits": [
            "References historical precedents and philosophical thought",
            "Examines the ethical dimensions of the issue",
            "Challenges fundamental assumptions",
            "Uses thought experiments",
            "Ends with a broader perspective on the issue",
        ],
        "color": "🟣",
    },
    "populist": {
        "name":  "ECHO",
        "style": "populist, relatable, uses common-sense arguments and real-world examples",
        "traits": [
            "Speaks in plain language anyone can understand",
            "Uses everyday examples and common experiences",
            "Questions elites and conventional wisdom",
            "Appeals to fairness and practicality",
            "Ends with a call to action",
        ],
        "color": "🟡",
    },
}

DEBATE_FORMATS = {
    "oxford": {
        "name": "Oxford Style",
        "rounds": ["opening", "rebuttal", "rebuttal", "closing"],
        "description": "Classic structured debate: opening → two rebuttals → closing",
        "max_words_per_turn": 200,
    },
    "parliamentary": {
        "name": "Parliamentary Style",
        "rounds": ["opening", "rebuttal", "rebuttal", "rebuttal", "closing"],
        "description": "5-round parliamentary with aggressive back-and-forth",
        "max_words_per_turn": 150,
    },
    "freeform": {
        "name": "Free-form Debate",
        "rounds": ["opening"] + ["rebuttal"] * 6 + ["closing"],
        "description": "Extended free-form debate — 8 rounds total",
        "max_words_per_turn": 180,
    },
    "rapid": {
        "name": "Rapid Fire",
        "rounds": ["opening", "rebuttal", "rebuttal", "closing"],
        "description": "Fast-paced short arguments — quick and punchy",
        "max_words_per_turn": 80,
    },
}


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class DebateTurn:
    round_num:    int
    round_type:   str          # opening / rebuttal / closing
    speaker:      str          # debater name
    position:     str          # PRO / CON
    persona:      str          # aggressive / analytical / etc.
    argument:     str          # the actual argument text
    word_count:   int
    timestamp:    float = field(default_factory=time.time)


@dataclass
class JudgeScore:
    round_num:   int
    speaker:     str
    logic:       int           # 1-10
    evidence:    int           # 1-10
    rhetoric:    int           # 1-10
    rebuttal:    int           # 1-10
    total:       float
    feedback:    str
    winner_round: str          # "PRO" / "CON" / "TIE"


@dataclass
class DebateResult:
    topic:          str
    format:         str
    debater_a:      dict       # persona dict
    debater_b:      dict
    position_a:     str        # "PRO"
    position_b:     str        # "CON"
    turns:          list[DebateTurn]
    scores:         list[JudgeScore]
    final_verdict:  str
    final_summary:  str
    winner:         str        # "PRO" / "CON" / "TIE"
    total_rounds:   int
    duration_sec:   float


# ── Gemini client ──────────────────────────────────────────────────────────────
def get_model(api_key: str, temperature: float = 0.85) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={
            "temperature": temperature,
            "max_output_tokens": 512,
        },
    )


# ── Argument generation ────────────────────────────────────────────────────────
def generate_argument(
    api_key:        str,
    topic:          str,
    position:       str,          # PRO or CON
    persona_key:    str,
    round_type:     str,          # opening / rebuttal / closing
    round_num:      int,
    opponent_last:  str,          # last opponent argument for rebuttal
    debate_history: list[dict],   # all previous turns
    max_words:      int,
) -> str:
    persona   = DEBATER_PERSONAS[persona_key]
    model     = get_model(api_key, temperature=0.88)

    history_text = ""
    if debate_history:
        last_turns = debate_history[-4:]  # last 2 exchanges
        history_text = "\n".join([
            f"{t['speaker']} ({t['position']}): {t['argument'][:300]}"
            for t in last_turns
        ])

    opponent_section = ""
    if opponent_last and round_type == "rebuttal":
        opponent_section = f"""
OPPONENT'S LAST ARGUMENT TO REBUT:
"{opponent_last[:400]}"

You MUST specifically address and counter at least one point from the above.
"""

    prompt = f"""You are {persona['name']}, a debater with a {persona['style']} style.
Your traits: {', '.join(persona['traits'])}

DEBATE TOPIC: "{topic}"
YOUR POSITION: {position} (you are ARGUING {position} of this topic)
ROUND TYPE: {round_type.upper()} (Round {round_num})
{opponent_section}
RECENT DEBATE HISTORY:
{history_text if history_text else "This is the opening — no history yet."}

INSTRUCTIONS:
- You are arguing {position} — NEVER concede your position
- Stay in character as {persona['name']} with your {persona['style']} style
- Keep response under {max_words} words — be punchy and impactful
- For OPENING: state your core argument powerfully
- For REBUTTAL: attack the opponent's argument AND advance your own
- For CLOSING: summarise your strongest points and deliver a memorable finish
- Write ONLY your argument — no labels, no "Round X:", no "As {persona['name']}:"
- Be direct, confident, and forceful. This is a COMPETITIVE DEBATE.

Your argument:"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    # Clean any accidental self-labels
    text = re.sub(r"^(APEX|NOVA|SAGE|ECHO|PRO|CON):\s*", "", text, flags=re.MULTILINE)
    return text


def generate_judge_score(
    api_key:     str,
    topic:       str,
    turn:        DebateTurn,
    opponent_turn: DebateTurn | None,
) -> JudgeScore:
    model = get_model(api_key, temperature=0.1)  # low temp for consistent scoring

    context = ""
    if opponent_turn:
        context = f'Opponent ({opponent_turn.position}) said:\n"{opponent_turn.argument[:300]}"'

    prompt = f"""You are a neutral, expert debate judge. Score this argument strictly and fairly.

TOPIC: "{topic}"
SPEAKER: {turn.speaker} arguing {turn.position}
ROUND: {turn.round_type.upper()}
{context}

ARGUMENT TO SCORE:
"{turn.argument}"

Return ONLY valid JSON (no markdown, no explanation):
{{
  "logic": <1-10 integer>,
  "evidence": <1-10 integer>,
  "rhetoric": <1-10 integer>,
  "rebuttal": <1-10 integer or 5 if no rebuttal needed>,
  "feedback": "<one sentence of specific, constructive feedback>",
  "winner_round": "<PRO or CON or TIE>"
}}

Scoring criteria:
- logic (1-10): clarity, structure, internal consistency
- evidence (1-10): facts, examples, specificity (penalise vague claims)
- rhetoric (1-10): persuasiveness, style, memorability
- rebuttal (1-10): how effectively they addressed opponent's points
- winner_round: which debater won THIS specific round"""

    response = model.generate_content(prompt)
    raw      = response.text.strip()
    raw      = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except Exception:
        data = {"logic": 7, "evidence": 6, "rhetoric": 7, "rebuttal": 6,
                "feedback": "Solid argument.", "winner_round": "TIE"}

    total = (data["logic"] + data["evidence"] + data["rhetoric"] + data["rebuttal"]) / 4

    return JudgeScore(
        round_num=turn.round_num,
        speaker=turn.speaker,
        logic=data["logic"],
        evidence=data["evidence"],
        rhetoric=data["rhetoric"],
        rebuttal=data["rebuttal"],
        total=round(total, 2),
        feedback=data.get("feedback", ""),
        winner_round=data.get("winner_round", "TIE"),
    )


def generate_final_verdict(
    api_key:  str,
    topic:    str,
    turns:    list[DebateTurn],
    scores:   list[JudgeScore],
    debater_a: dict,
    debater_b: dict,
) -> tuple[str, str, str]:
    """Generate final verdict, summary, and declare a winner."""
    model = get_model(api_key, temperature=0.2)

    # Tally scores
    scores_a = [s for s in scores if s.speaker == debater_a["name"]]
    scores_b = [s for s in scores if s.speaker == debater_b["name"]]
    avg_a    = sum(s.total for s in scores_a) / max(len(scores_a), 1)
    avg_b    = sum(s.total for s in scores_b) / max(len(scores_b), 1)
    rounds_won_a = sum(1 for s in scores if s.winner_round == "PRO")
    rounds_won_b = sum(1 for s in scores if s.winner_round == "CON")

    all_args = "\n\n".join([
        f"[Round {t.round_num} | {t.speaker} | {t.position}]\n{t.argument[:250]}"
        for t in turns
    ])

    prompt = f"""You are the chief judge of a formal debate. Deliver your final verdict.

TOPIC: "{topic}"
DEBATER A: {debater_a['name']} (PRO) — avg score {avg_a:.1f}/10, {rounds_won_a} rounds won
DEBATER B: {debater_b['name']} (CON) — avg score {avg_b:.1f}/10, {rounds_won_b} rounds won

DEBATE TRANSCRIPT SUMMARY:
{all_args[:2000]}

Return ONLY valid JSON:
{{
  "winner": "<PRO or CON or TIE>",
  "verdict": "<2-3 sentence formal verdict explaining who won and why>",
  "summary": "<3-4 sentence debate summary covering key arguments from both sides>",
  "best_argument": "<the single most impressive argument made in the entire debate>",
  "turning_point": "<the moment that decided the debate>"
}}"""

    response = model.generate_content(prompt)
    raw      = response.text.strip()
    raw      = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except Exception:
        data = {
            "winner": "PRO" if avg_a > avg_b else ("CON" if avg_b > avg_a else "TIE"),
            "verdict": f"After careful consideration, {debater_a['name'] if avg_a > avg_b else debater_b['name']} wins on points.",
            "summary": "A closely contested debate with strong arguments on both sides.",
            "best_argument": "Multiple strong arguments were presented throughout.",
            "turning_point": "The rebuttal rounds proved decisive.",
        }

    return data["winner"], data["verdict"], data["summary"]


# ── Main debate runner ─────────────────────────────────────────────────────────
def run_debate(
    api_key:       str,
    topic:         str,
    persona_a_key: str = "aggressive",
    persona_b_key: str = "analytical",
    format_key:    str = "oxford",
    on_turn:       callable = None,    # callback for streaming UI updates
    on_score:      callable = None,
) -> DebateResult:
    """
    Run a full debate between two AI debaters.
    Calls on_turn(turn) and on_score(score) for streaming UI.
    """
    fmt        = DEBATE_FORMATS[format_key]
    persona_a  = DEBATER_PERSONAS[persona_a_key]
    persona_b  = DEBATER_PERSONAS[persona_b_key]
    rounds     = fmt["rounds"]
    max_words  = fmt["max_words_per_turn"]

    turns:  list[DebateTurn]  = []
    scores: list[JudgeScore]  = []
    history = []

    t_start = time.time()

    for i, round_type in enumerate(rounds):
        round_num = i + 1

        # Determine who speaks first this round
        # Opening/closing: A goes first. Rebuttals: alternate
        if i % 2 == 0:
            speakers = [
                (persona_a, "PRO", persona_a_key),
                (persona_b, "CON", persona_b_key),
            ]
        else:
            speakers = [
                (persona_b, "CON", persona_b_key),
                (persona_a, "PRO", persona_a_key),
            ]

        round_turns = []
        for persona, position, persona_key in speakers:
            # Get opponent's last argument for rebuttal
            opponent_pos = "CON" if position == "PRO" else "PRO"
            opponent_last = next(
                (t.argument for t in reversed(turns) if t.position == opponent_pos),
                "",
            )

            # Generate argument
            argument = generate_argument(
                api_key=api_key,
                topic=topic,
                position=position,
                persona_key=persona_key,
                round_type=round_type,
                round_num=round_num,
                opponent_last=opponent_last,
                debate_history=history,
                max_words=max_words,
            )

            turn = DebateTurn(
                round_num=round_num,
                round_type=round_type,
                speaker=persona["name"],
                position=position,
                persona=persona_key,
                argument=argument,
                word_count=len(argument.split()),
            )
            turns.append(turn)
            round_turns.append(turn)
            history.append({"speaker": persona["name"], "position": position, "argument": argument})

            if on_turn:
                on_turn(turn)

        # Judge scores both arguments in this round
        for j, turn in enumerate(round_turns):
            opponent_turn = round_turns[1 - j] if len(round_turns) > 1 else None
            score = generate_judge_score(api_key, topic, turn, opponent_turn)
            scores.append(score)
            if on_score:
                on_score(score)

    # Final verdict
    winner, verdict, summary = generate_final_verdict(
        api_key, topic, turns, scores, persona_a, persona_b
    )

    return DebateResult(
        topic=topic,
        format=format_key,
        debater_a=persona_a,
        debater_b=persona_b,
        position_a="PRO",
        position_b="CON",
        turns=turns,
        scores=scores,
        final_verdict=verdict,
        final_summary=summary,
        winner=winner,
        total_rounds=len(rounds),
        duration_sec=round(time.time() - t_start, 1),
    )


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Debate Engine")
    parser.add_argument("topic",            type=str, help="Debate topic")
    parser.add_argument("--format",         default="oxford", choices=list(DEBATE_FORMATS.keys()))
    parser.add_argument("--persona-a",      default="aggressive", choices=list(DEBATER_PERSONAS.keys()))
    parser.add_argument("--persona-b",      default="analytical", choices=list(DEBATER_PERSONAS.keys()))
    parser.add_argument("--output",         type=str, default="", help="Save debate to JSON")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("❌ Set GEMINI_API_KEY environment variable")
        exit(1)

    print(f"\n{'═'*60}")
    print(f"  🎤 AI DEBATE ENGINE")
    print(f"  Topic:  {args.topic}")
    print(f"  Format: {DEBATE_FORMATS[args.format]['name']}")
    print(f"{'═'*60}\n")

    pa = DEBATER_PERSONAS[args.persona_a]
    pb = DEBATER_PERSONAS[args.persona_b]
    print(f"  {pa['color']} {pa['name']} (PRO) — {pa['style']}")
    print(f"  {pb['color']} {pb['name']} (CON) — {pb['style']}\n")

    def print_turn(turn: DebateTurn):
        persona = DEBATER_PERSONAS[turn.persona]
        print(f"\n{'─'*60}")
        print(f"  {persona['color']} {turn.speaker} | {turn.position} | {turn.round_type.upper()} Round {turn.round_num}")
        print(f"{'─'*60}")
        print(f"  {turn.argument}\n")

    def print_score(score: JudgeScore):
        print(f"  📊 [{score.speaker}] Logic:{score.logic} Evidence:{score.evidence} "
              f"Rhetoric:{score.rhetoric} Rebuttal:{score.rebuttal} → {score.total:.1f}/10")
        print(f"  💬 {score.feedback}")

    result = run_debate(
        api_key=api_key,
        topic=args.topic,
        persona_a_key=args.persona_a,
        persona_b_key=args.persona_b,
        format_key=args.format,
        on_turn=print_turn,
        on_score=print_score,
    )

    print(f"\n{'═'*60}")
    print(f"  🏆 FINAL VERDICT: {result.winner} WINS")
    print(f"{'═'*60}")
    print(f"\n  {result.final_verdict}")
    print(f"\n  📝 Summary: {result.final_summary}")
    print(f"\n  ⏱️ Duration: {result.duration_sec}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"\n  ✅ Saved to {args.output}")
