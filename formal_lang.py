# -----------------------
# Tribal conversations
# -----------------------
conversations = [
    [
        "zuma keta rilo",
        "rilo pona suva",
        "suva keta doro",
        "doro zuma pini",
    ],
    [
        "mira tolu sako",
        "sako neri vipa",
        "vipa tolu rani",
        "rani mira doku",
    ],
    [
        "pavo lira kuni",
        "kuni meko zera",
        "zera lira hato",
        "hato pavo nuli",
    ],
    [
        "tari moku sena",
        "sena jeko rumi",
        "rumi moku pela",
        "pela tari nado",
    ],
    [
        "nema suki rako",
        "rako bimi tanu",
        "tanu suki velo",
        "velo nema piri",
    ],
    [
        "janu kelo sili",
        "sili ramo teku",
        "teku kelo nari",
        "nari janu peka",
    ],
    [
        "feko rina melo",
        "melo tisa buro",
        "buro rina sedi",
        "sedi feko lani",
    ],
    [
        "duni pago tika",
        "tika mero suli",
        "suli pago renu",
        "renu duni lako",
    ],
    [
        "vona temi paku",
        "paku lera simo",
        "simo temi jaku",
        "jaku vona reli",
    ],
    [
        "beka rumi tono",
        "tono lesi mifa",
        "mifa rumi zoki",
        "zoki beka nalo",
    ],
    [
        "cari meno tupa",
        "tupa jelu rasi",
        "rasi meno dika",
        "dika cari lupo",
    ],
    [
        "sepi karo timo",
        "timo nevi laku",
        "laku karo sime",
        "sime sepi rudo",
    ],
    [
        "gito rafa lani",
        "lani peko rida",
        "rida rafa moki",
        "moki gito nera",
    ],
    [
        "pila mode renu",
        "renu dabe tuku",
        "tuku mode sani",
        "sani pila meku",
    ],
    [
        "tabe firo nali",
        "nali vemo suta",
        "suta firo jeni",
        "jeni tabe rolu",
    ],
    [
        "mado kesi rine",
        "rine tosa peki",
        "peki kesi jaro",
        "jaro mado vesi",
    ],
    [
        "lubi seno taro",
        "taro peni kima",
        "kima seno duro",
        "duro lubi mete",
    ],
    [
        "savi piro jalo",
        "jalo mevi tono",
        "tono piro nese",
        "nese savi ruka",
    ],
    [
        "ruka temu nalo",
        "nalo vasi jepo",
        "jepo temu rini",
        "rini ruka mepa",
    ],
    [
        "puna kedi raso",
        "raso miti leko",
        "leko kedi saro",
        "saro puna vike",
    ],
    [
        "nemi joru lita",
        "lita sevo rupi",
        "rupi joru kela",
        "kela nemi sado",
    ],
    [
        "bira tami selo",
        "selo neri jaku",
        "jaku tami leto",
        "leto bira rumi",
    ],
    [
        "tupa rini selo",
        "selo davi pemo",
        "pemo rini kuto",
        "kuto tupa lemi",
    ],
    [
        "vela sumi pado",
        "pado meku sini",
        "sini sumi ravo",
        "ravo vela jeni",
    ],
    [
        "tori mika senu",
        "senu dopa liri",
        "liri mika nuvo",
        "nuvo tori seka",
    ],
]

confused_responses = [
    "mira keta rina"
]

lexicon = sorted(w for convo in conversations for sent in convo for w in sent.split())


# -*- coding: utf-8 -*-
# Requirements: pip install openai tqdm

import os
import time
import random
import logging
from openai import OpenAI
from tqdm import tqdm

# -------------------------------------------------
# Assumes you already define:
#   conversations: list[list[str]]
#   confused_responses: list[str]
# -------------------------------------------------

# -----------------------
# Gorilla language helpers
# -----------------------
def find_sentence_position(sentence):
    """Return (conv_index, position) if valid, else None."""
    for ci, conv in enumerate(conversations):
        if sentence in conv:
            return ci, conv.index(sentence)
    return None

def gorilla_bot_reply(gemini_msg):
    """
    Gorilla Bot reply logic:
    - If Gemini said a valid sentence (and not last) → reply 'hanu' + mapped next.
    - If Gemini said the last sentence in conv → reply random start (no hanu).
    - If invalid → confused response (no hanu).
    """
    pos = find_sentence_position(gemini_msg)
    if pos is None:
        return random.choice(confused_responses), None  # invalid reset

    conv_index, idx = pos
    conv = conversations[conv_index]

    if idx < len(conv) - 1:
        return "hanu " + conv[idx + 1], (conv_index, idx + 1)
    else:
        new_conv = random.choice(conversations)
        return new_conv[0], (conversations.index(new_conv), 0)

# -----------------------
# OpenRouter / OpenAI setup
# -----------------------
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

FAST_MODELS = [
    # "deepseek/deepseek-r1-0528:free",
    # "meta-llama/llama-3.3-8b-instruct:free",
    # "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    # "anthropic/claude-3.5-haiku"
    # "openai/gpt-3.5-turbo-0125",
]

# Suggested round lengths for testing
ROUND_LENGTHS = [200, 200, 200, 200, 200, 200, 200, 200]

# -----------------------
# TQDM-aware logger setup
# -----------------------
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

logger = logging.getLogger("exp")
logger.setLevel(logging.INFO)
for h in list(logger.handlers):
    logger.removeHandler(h)
_tqdm_handler = TqdmLoggingHandler()
_tqdm_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_tqdm_handler)

# -----------------------
# Core experiment function (one model × one max_rounds)
# -----------------------
def gorilla_vs_gemini(MODEL, max_rounds, run_id, total_runs, global_pbar):
    """
    Run Gorilla Bot vs LLM for a single model and max_rounds setting.
    Saves:
      - Raw transcript to logs/{MODEL}_round{max_rounds}_{run_id}.txt
      - Metrics to metrics/{MODEL}_metrics.txt (append mode)
    Shows:
      - Per-run progress bar with live metric postfix
      - Per-round metric line via tqdm-aware logger
    """
    os.makedirs("logs", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    transcript = []
    successful_conversations = 0
    rounds = 0

    # Metrics storage
    metrics = {
        "valid_turns": 0,
        "total_turns": 0,
        "conversation_successes": 0,
        "conversation_attempts": 0,
        "adjacency_matches": 0,
        "adjacency_total": 0,
        # Recovery after negative feedback (immediate next turn only)
        "feedback_recoveries": 0,
        "feedback_opportunities": 0,
        "awaiting_immediate_recovery": False,  # <- new flag
        # Time to first positive feedback
        "first_valid_round": None,              # <- new
        "turns_taken": 0
    }

    # System prompt for this run
    chat_history = [
        {"role": "system", "content": "You are a language expert talking with a primal tribe through chat. The tribe does not understand any language apart from their own. The tribal language does notinclude any punctuation or uppercase letters or numbers. Your goal is to understand and reply back to the tribe accordingly. Whenever you reply with a valid sentence in the language, the tribe starts their reply with 'hanu'. When you say an invalid sentence, they reply 'mira keta rina'. Do not start your replies with 'hanu'. You are supposed to have three successful conversations with the tribe. Reply back in only the tribal language, a response you deem appropriate. Do not reply with anything else. You interact with the tribe from now."}
    ]

    # Start with a random conversation beginning
    current_conv = random.choice(conversations)
    gorilla_msg = current_conv[0]
    current_context = (conversations.index(current_conv), 0)
    prev_invalid = False

    # Per-run progress bar
    with tqdm(
        total=max_rounds,
        desc=f"{MODEL} | R={max_rounds} | Run={run_id}",
        ncols=100,
        leave=False,
        position=1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ) as run_pbar:

        while successful_conversations < 3 and rounds < max_rounds:
            rounds += 1
            metrics["turns_taken"] += 1

            # Tribal bot speaks
            transcript.append(f"Tribal Bot: {gorilla_msg}")
            chat_history.append({"role": "user", "content": gorilla_msg})

            # LLM replies
            start = time.time()
            response = client.chat.completions.create(
                model=MODEL,
                messages=chat_history
            )
            end = time.time()

            gemini_msg = response.choices[0].message.content.strip().lower()
            chat_history.append({"role": "assistant", "content": gemini_msg})
            transcript.append(f"{MODEL}: {gemini_msg}   (took {end-start:.2f}s)")

            # Update totals
            metrics["total_turns"] += 1

            # Validity and mapping
            pos = find_sentence_position(gemini_msg)
            if pos is not None:
                # First valid turn → time to first positive feedback
                if metrics["first_valid_round"] is None:
                    metrics["first_valid_round"] = rounds

                metrics["valid_turns"] += 1
                conv_index, idx = pos

                # adjacency check against the bot's last msg
                metrics["adjacency_total"] += 1
                if any(word in gemini_msg for word in gorilla_msg.split()):
                    metrics["adjacency_matches"] += 1

                # correct next sentence inside the same conversation
                if current_context and conv_index == current_context[0] and idx == current_context[1] + 1:
                    if idx == len(conversations[conv_index]) - 1:
                        metrics["conversation_successes"] += 1
                        metrics["conversation_attempts"] += 1

                # Immediate recovery check: valid right after a single invalid
                if metrics["awaiting_immediate_recovery"]:
                    metrics["feedback_recoveries"] += 1
                    metrics["awaiting_immediate_recovery"] = False

            else:
                # Invalid turn → this counts as a new recovery opportunity
                metrics["conversation_attempts"] += 1
                # We only count the immediate next turn for recovery.
                # If the next turn is valid → recovery++. If invalid again → no recovery and we stop waiting.
                metrics["feedback_opportunities"] += 1
                metrics["awaiting_immediate_recovery"] = True

            # Recovery: a valid turn right after an invalid
            if prev_invalid is False and metrics["feedback_opportunities"] > metrics["feedback_recoveries"]:
                metrics["feedback_recoveries"] += 1

            # Tribal bot replies to the LLM
            gorilla_msg, current_context = gorilla_bot_reply(gemini_msg)

            # If we were awaiting an immediate recovery and got another invalid,
            # we already counted that opportunity; stop waiting now.
            if metrics["awaiting_immediate_recovery"] and pos is None:
                metrics["awaiting_immediate_recovery"] = False

            # ---------- per-round progress and metrics ----------
            # compute intermediate metrics
            TVR = metrics["valid_turns"] / metrics["total_turns"] if metrics["total_turns"] else 0.0
            CSR = (
                metrics["conversation_successes"] / metrics["conversation_attempts"]
                if metrics["conversation_attempts"] else 0.0
            )
            AC = (
                metrics["adjacency_matches"] / metrics["adjacency_total"]
                if metrics["adjacency_total"] else 0.0
            )
            FR = (
                metrics["feedback_recoveries"] / metrics["feedback_opportunities"]
                if metrics["feedback_opportunities"] else 0.0
            )
            CE = metrics["turns_taken"]
            TTFK = metrics["first_valid_round"] if metrics["first_valid_round"] is not None else -1  # or None

            # update per-run bar and postfix
            run_pbar.set_postfix(TVR=f"{TVR:.2f}", CSR=f"{CSR:.2f}", AC=f"{AC:.2f}", FR=f"{FR:.2f}", CE=CE, TTFK = TTFK)
            run_pbar.update(1)


    # Final metrics for the run
    TVR = metrics["valid_turns"] / metrics["total_turns"] if metrics["total_turns"] else 0.0
    CSR = (
        metrics["conversation_successes"] / metrics["conversation_attempts"]
        if metrics["conversation_attempts"] else 0.0
    )
    AC = (
        metrics["adjacency_matches"] / metrics["adjacency_total"]
        if metrics["adjacency_total"] else 0.0
    )
    FR = (
        metrics["feedback_recoveries"] / metrics["feedback_opportunities"]
        if metrics["feedback_opportunities"] else 0.0
    )
    CE = metrics["turns_taken"]
    # Time to First Positive Feedback (round index when first valid occurred)
    TTFK = metrics["first_valid_round"] if metrics["first_valid_round"] is not None else -1  # or None

    # Save transcript
    log_file = f"logs/{MODEL.replace('/','_')}_round{max_rounds}_{run_id}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        for line in transcript:
            f.write(line + "\n")

    # Save metrics
    metric_file = f"metrics/{MODEL.replace('/','_')}_metrics.txt"
    with open(metric_file, "a", encoding="utf-8") as f:
        f.write(f"Run {run_id}, Rounds={max_rounds}\n")
        f.write(f"Turn Validity Rate (TVR): {TVR:.2f}\n")
        f.write(f"Conversation Success Rate (CSR): {CSR:.2f}\n")
        f.write(f"Adjacency Compliance (AC): {AC:.2f}\n")
        f.write(f"Feedback Responsiveness (FR): {FR:.2f}\n")
        f.write(f"Completion Efficiency (CE): {CE}\n")
        f.write(f"Time to First Positive Feedback (TTFK): {TTFK}\n") 
        f.write("-" * 40 + "\n")

    # Log summary and advance global bar
    logger.info(
        f"✅ Finished Run {run_id}/{total_runs} | Model={MODEL} | Rounds={max_rounds} | "
        f"TVR={TVR:.2f}, CSR={CSR:.2f}, AC={AC:.2f}, FR={FR:.2f}, TTFK={TTFK}, CE={CE}"
    )

    logger.info(f"Raw transcript saved → {log_file}")
    logger.info(f"Metrics appended → {metric_file}")
    global_pbar.update(1)

# -----------------------
# Run all tests with a global progress bar
# -----------------------
def run_all():
    total_runs = len(FAST_MODELS) * len(ROUND_LENGTHS)
    run_id = 1

    logger.info(
        f"Starting experiments: {total_runs} total runs "
        f"({len(FAST_MODELS)} models × {len(ROUND_LENGTHS)} round lengths)"
    )

    with tqdm(
        total=total_runs,
        desc="Overall Progress",
        ncols=100,
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ) as global_pbar:
        for model in FAST_MODELS:
            for rounds in ROUND_LENGTHS:
                logger.info(
                    f"▶️ Starting Run {run_id}/{total_runs} "
                    f"(Model={model}, MaxRounds={rounds})"
                )
                gorilla_vs_gemini(model, rounds, run_id, total_runs, global_pbar)
                run_id += 1

    logger.info("🎉 All experiments completed!")

# ---------------
# Entry point
# ---------------
if __name__ == "__main__":
    run_all()

