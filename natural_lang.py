import pandas as pd
from collections import defaultdict
import re
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class ConlangTranslator:
    def __init__(self, cl_to_en, en_to_cl, api_key: str):
        """
        cl_to_en: dict mapping conlang → english
        en_to_cl: dict mapping english → [conlang words]
        api_key : your OpenRouter API key
        """
        self.cl_to_en = cl_to_en
        self.en_to_cl = en_to_cl
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def translate(self, sentence: str, direction: str) -> str:
        # --- Step 1. Extract non-verbal gesture ---
        gesture_match = re.match(r"\((.*?)\)\s*(.*)", sentence.strip())
        if gesture_match:
            non_verbal = gesture_match.group(1).strip()
            verbal_part = gesture_match.group(2).strip()
        else:
            non_verbal = ""
            verbal_part = sentence.strip()

        if direction == "cl_to_en":
            # --- Step 2. Word mapping first ---
            mapped_tokens = []
            for token in verbal_part.split():
                if token in self.cl_to_en:
                    mapped_tokens.append(self.cl_to_en[token])
                else:
                    mapped_tokens.append("[unknown]")
            mapped_sentence = " ".join(mapped_tokens)

            # --- Step 3. Syntax reorder by LLM ---
            reordered_sentence = self.reorder_with_llm(mapped_sentence, direction, non_verbal)

        elif direction == "en_to_cl":
            # --- Step 2. Syntax reorder by LLM first ---
            reordered_sentence = self.reorder_with_llm(verbal_part, direction, non_verbal)

            # --- Step 3. Word mapping ---
            mapped_tokens = []
            for token in reordered_sentence.split():
                if token in self.en_to_cl:
                    mapped_tokens.append(self.en_to_cl[token][0])
                else:
                    mapped_tokens.append("[unknown]")
            reordered_sentence = " ".join(mapped_tokens)

        else:
            raise ValueError("Invalid direction. Use 'cl_to_en' or 'en_to_cl'.")

        # --- Step 4. Add non-verbal gesture back ---
        if non_verbal:
            reordered_sentence = f"({non_verbal}) {reordered_sentence}"

        return reordered_sentence

    def reorder_with_llm(self, text: str, direction: str, non_verbal: str) -> str:
        """
        Calls OpenRouter LLM to reorder sentence syntax using the provided rules.
        Appends the non-verbal gesture after receiving the reordered sentence.
        """

        prompt = f"""
You are a language model trained to translate between English and a constructed language (conlang) that follows a specific syntactic structure.

Your task is to reorder words between the two languages without changing or inventing any words — only their positions according to the rules below.

Conversion Rules
1. Sentence Type
   - If the input is in English, convert it to conlang order.
   - If the input is in conlang structure, convert it to English order.

2. Conlang Sentence Structure
   [Marker] [Neg] Verb Object Subject

3. Word Structures
   Verb: [Asp] [Root Verb]
   Noun: [Noun Phrase] [Post Position]
   Noun Phrase (NP): [Quantifier] [Number] [Root Noun] [Modifier] [Relative Clause]
   Relative Clause: [Desc] V O S

4. Word Role Definitions
   Marker → indicates sentence type (declarative, interrogative, imperative).
   Neg → a single word expressing negation.
   Asp → an adverb placed before the root verb.
   Desc → a marker indicating the start of a relative clause.

5. Marker insertion (only when translation is from english to conlang)
   If the sentence is declarative, add "DECL" at the start of the sentence
   If the sentence is interrogative, add "INTR" at the start of the sentence
   If the sentence is imperative, add "IMPR" at the start of the sentence

Output Instructions
- Only if the input text contains of only [unknown] words, let it be as it is.
- Do not translate or change any words.
- Do not add, remove, or modify grammar markers.
- Simply reorder the words according to the above structures.
- Reply back with just the reordered sentence.

Direction: {direction}
Input: {text}
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "openai/gpt-4o-mini",  # swap for another model if you prefer
            "messages": [
                {"role": "system", "content": "You are a precise syntactic reordering engine."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

        result = response.json()
        reordered = result["choices"][0]["message"]["content"].strip()

        return reordered

def translated_english_response(foreigner_message: str, api_key: str) -> str:
    """
    Simulates a tribal response to a foreigner's message.
    Calls an LLM via OpenRouter API with the tribal role prompt.

    Parameters
    ----------
    foreigner_message : str
        The message produced by the translator (input from foreigner).
    api_key : str
        Your OpenRouter API key.

    Returns
    -------
    str : Tribal-style response in primitive conlang.
    """

    prompt = f"""
You are a tribesperson living deep within an ancient forest, part of a small, isolated tribe that speaks its own unique language. One day, you encounter a foreigner who clearly does not belong to your land. Your task is to respond to him as a member of your tribe would — cautiously, curiously, and with limited intelligence. You should:

- Reply back with only a single sentence in a broken English using only the words from the set below. Do not use words outside this set. The sentence does not have to be grammatically correct.

Set of available words - completed, habitual, potential, nearly finished, interrupt, resume, temporary, permanent, sudden, gradual, repetitive, single, preparatory, delayed, simultaneous, sequential, accidental, intentional, ongoing, seasonal, once, everyday, incomplete, exhaustive, swift, slow, uncertain, inevitable, enduring, fading, meaning, and, but, because, if, so, or, not, big, small, long, short, round, straight, bent, sharp, hard, soft, smooth, rough, good, bad, useful, harmful, more, less, full, empty, bright, dark, hot, cold, fast, slow, heavy, light, old, new, animate, inanimate, natural, abstract, body, head, hand, eye, ear, mouth, heart, foot, child, elder, friend, stranger, animal, tree, water, sky, sun, moon, star, mountain, river, stone, fire, house, tool, rope, stick, boat, food, group, place, path, road, village, field, forest, sky-thing, earth-thing, time, day, night, person, thing, spirit, wind, rain, cloud, sound, earth, liquid, container, plant, artifact, event, zero, one, two, three, four, five, six, seven, eight, nine, in, on, with, from, before, after, near, far, inside, outside, above, below, between, around, through, toward, against, about, during, without, all, some, none, few, many, each, every, part, move, see, speak, touch, hear, give, take, make, break, join, rise, fall, enter, exit, run, stop, flow, eat, drink, think, push, pull, carry, throw, catch, cut, grow, burn, sleep, dream, remember, forget, feel, sing, call, look, listen, smell, taste, walk, stand, sit, lie, begin, end, play, fight

- Optionally include non-verbal actions or reactions in parentheses (e.g., (points at him), (tilts head curiously)).
- Your non-verbal reach should be appended to the start of the sentence in english within parentheses.
- All words must be in lowercase.
- Do not use any punctuation.

If the input contains only [unknown] words, it means that you do not understand what the foreigner is saying verbally. You will now respond to the foreigner.

The foreigner says: {foreigner_message}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-4o-mini",   # swap to another OpenRouter model if needed
        "messages": [
            {"role": "system", "content": "You are a tribesperson."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

    result = response.json()
    tribal_reply = result["choices"][0]["message"]["content"].strip()
    return tribal_reply

def toffi(translator: ConlangTranslator, foreigner_message: str, api_key: str = "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") -> dict:
    """
    Executes a full conversation turn in the pipeline:
    1. Conlang → English translation
    2. Reorder with LLM (cl_to_en)
    3. Tribal response in English
    4. Reorder with LLM (en_to_cl)
    5. English → Conlang translation

    Parameters
    ----------
    translator : ConlangTranslator
        Translator instance with loaded vocab + API key
    foreigner_message : str
        Message from foreigner in conlang
    api_key : str
        OpenRouter API key

    Returns
    -------
    dict : results of each stage
    {
      "cl_to_en": str,
      "tribal_english_response": str,
      "final_conlang": str
    }
    """

    # Step 1. Conlang → English
    cl_to_en = translator.translate(foreigner_message, direction="cl_to_en")

    # Step 2. Tribal English response
    tribal_english_response = translated_english_response(cl_to_en, api_key)

    # Step 3. English → Conlang
    final_conlang = translator.translate(tribal_english_response, direction="en_to_cl")

    # Remove all [unknown] tokens
    final_conlang = " ".join([tok for tok in final_conlang.split() if tok != "[unknown]"])
    return {
        "cl_to_en": cl_to_en,
        "tribal_english_response": tribal_english_response,
        "final_conlang": final_conlang,
    }

def test_LLM(api_key: str, history: list[str], model_name: str) -> str:
    """
    Simulates a language expert trying to communicate with a tribe
    to achieve three survival objectives.
    Uses the full conversational history for context.

    Parameters
    ----------
    api_key : str
        Your OpenRouter API key.
    history : list of str
        Full conversational history so far (in English or mixed).
        Each element is a turn in the dialogue.

    Returns
    -------
    str : The next tribal-style sentence produced by the expert.
    """

    # Base instructions
    base_prompt = """
You are a language expert who has become lost in a dense forest and encounters an unknown tribe. 
Your mission is to successfully communicate with the tribe in their tribal language to achieve the following three objectives, in order:

1. Ask for food and water.
2. Ask if you can stay for the night.
3. Ask for directions to leave the forest.

Response Format (strict):
- Your response is in the format: (non_verbal_gesture) verbal_sentence
- If you wish to specify a non-verbal gesture, you can describe it at the start of your sentence within parentheses.
- You are allowed to describe the non-verbal gesture but the intention of the gesture must not be specified. For example, (pointing to mouth) is valid, but (pointing to mouth to indicate hungry) is not valid.
- Follow this with only a single sentence strictly in the tribal language only.
- Reply back only in lowercase
- Do not use punctuation in verbal_sentence
- The non_verbal_gesture must be specified in English, whereas the verbal_sentence must be in the tribal language.


After each objective is completed, you will receive feedback confirming your success.
Continue the conversation based on the history.
"""

    # Format conversation history as a string
    history_str = "\n".join(history)

    full_prompt = f"{base_prompt}\n\nConversation so far:\n{history_str}\n\nYour next response:"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,  # or another OpenRouter model
        "messages": [
            {"role": "system", "content": "You are a stranded language expert."},
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0.7,
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

    result = response.json()
    expert_sentence = result["choices"][0]["message"]["content"].strip()
    return expert_sentence

#-------------------------------------
# Evaluation Block
#-------------------------------------

def categorize_sentence(sentence, word2cat):
    """Replace tokens in sentence with categories (skip [unknown])."""
    tokens = sentence.split()
    categories = []
    for tok in tokens:
        if tok == "[unknown]":
            continue
        categories.append(word2cat.get(tok, "UNK"))
    return categories

def reorder_cat_with_llm(categories, api_key):
    """Ask LLM to reorder categories into canonical conlang syntax."""
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    input_str = " ".join(categories)

    prompt = f"""
You are a syntax normalizer for a constructed language (conlang).
Reorder this sequence of syntactic categories into the correct canonical order.

Conlang Syntax:
- Sentence: [marker] [neg] Verb Object Subject
- Verb: [aspect] [verb-root]
- Noun Phrase: [quantifier] [number] [noun-root] [modifier] [rel-clause-connector?] [postpositional phrase]
- Relative Clause: [rel-clause-connector] V O S

Rules:
- Return only the reordered categories as a space-separated list.
- Do not add, remove, or translate.
- Use exactly the same categories, just reordered.

Input:
{input_str}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a precise syntax reordering engine."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    response = requests.post(base_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

    result = response.json()
    reordered = result["choices"][0]["message"]["content"].strip().split()
    return reordered

def levenshtein_distance(seq1, seq2):
    """Compute Levenshtein edit distance between two token lists."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def lexical_acquisition(transcript, conlang_vocab):
    """
    Evaluate lexical acquisition from a transcript of test_LLM responses.

    Parameters
    ----------
    transcript : list of str
        Sentences spoken by test_LLM.
    conlang_vocab : set
        Set of all valid conlang words from the lexicon.

    Returns
    -------
    unique_conlang_count : int
        Total number of unique conlang words used.
    overall_ratio : float
        Ratio of (total conlang words) / (total words) across transcript.
    avg_change : float
        Average change in ratio per sentence (trend).
    """

    total_conlang_words = 0
    total_words = 0
    unique_conlang = set()
    per_sentence_ratios = []

    for sent in transcript:
        tokens = sent.split()
        word_count = len(tokens)
        conlang_count = sum(1 for tok in tokens if tok in conlang_vocab)

        total_words += word_count
        total_conlang_words += conlang_count
        unique_conlang.update(tok for tok in tokens if tok in conlang_vocab)

        ratio = (conlang_count / word_count) if word_count > 0 else 0
        per_sentence_ratios.append(ratio * 100)  # percentage

    # 1. Total unique conlang words used
    unique_conlang_count = len(unique_conlang)

    # 2. Overall ratio across transcript
    overall_ratio = (total_conlang_words / total_words) * 100 if total_words > 0 else 0

    # 3. Average change in ratio across session
    if len(per_sentence_ratios) > 1:
        avg_change = (per_sentence_ratios[-1] - per_sentence_ratios[0]) / (len(per_sentence_ratios) - 1)
    else:
        avg_change = 0.0

    return unique_conlang_count, overall_ratio, avg_change

def syntax_evaluator(transcript, word2cat, api_key):
    """
    Evaluate syntax adherence using LLM-based reordering.

    Returns:
    --------
    strict_scores : list of float
        Token position match % per sentence.
    lev_scores : list of float
        Levenshtein similarity % per sentence.
    avg_change : float
        Average change in strict accuracy across session.
    weighted_strict : float
        Length-weighted strict score across transcript.
    weighted_lev : float
        Length-weighted Levenshtein score across transcript.
    """

    strict_scores, lev_scores = [], []
    weighted_strict_sum, weighted_lev_sum, total_len = 0.0, 0.0, 0

    for sent in transcript:
        cats = categorize_sentence(sent, word2cat)

        # Skip if sentence has <= 1 unique categories
        if len(set(cats)) <= 1:
            strict_scores.append(0.0)
            lev_scores.append(0.0)
            continue

        gold = reorder_cat_with_llm(cats, api_key)

        # Strict position match
        matches = sum(1 for i, tok in enumerate(cats) if i < len(gold) and tok == gold[i])
        strict_acc = (matches / max(len(cats), len(gold))) * 100
        strict_scores.append(strict_acc)

        # Levenshtein similarity
        dist = levenshtein_distance(cats, gold)
        lev_sim = (1 - dist / max(len(cats), len(gold))) * 100
        lev_scores.append(lev_sim)

        # Update weighted sums
        weighted_strict_sum += strict_acc * len(cats)
        weighted_lev_sum += lev_sim * len(cats)
        total_len += len(cats)

    # Avg change across session (strict scores)
    if len(strict_scores) > 1:
        avg_change = (strict_scores[-1] - strict_scores[0]) / (len(strict_scores) - 1)
    else:
        avg_change = 0.0

    # Weighted metrics (normalize by total token length)
    weighted_strict = weighted_strict_sum / total_len if total_len > 0 else 0.0
    weighted_lev = weighted_lev_sum / total_len if total_len > 0 else 0.0

    return strict_scores, lev_scores, avg_change, weighted_strict, weighted_lev

def evaluate_conversational_success(transcript, api_key):
    """
    Evaluates if test_LLM achieved the 3 objectives in conversation with the tribe.

    Parameters
    ----------
    transcript : str
        Full English-translated transcript of the conversation.
    api_key : str
        OpenRouter API key.

    Returns
    -------
    list[int] : [X, Y, Z] where each is 0 or 1
    """

    prompt = f"""
You are an evaluator. You will judge if the foreigner (test_LLM) successfully achieved 
three objectives in conversation with a tribe. 

Objectives:
1. Get the tribe to agree to give you food and water.
2. Get the tribe to agree to let you stay for the night.
3. Get directions to leave the forest.

Transcript of the full interaction (in English):
{transcript}

Instructions:
- Carefully read the transcript.
- Decide if each objective was achieved.
- Output strictly in the format [X, Y, Z], where each X, Y, Z is either 0 or 1.
- Do not explain, only return the list.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-4o-mini",  # or another evaluation model
        "messages": [
            {"role": "system", "content": "You are a strict evaluation engine."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

    result = response.json()
    eval_output = result["choices"][0]["message"]["content"].strip()

    # Parse output like "[1, 0, 1]"
    try:
        success_list = eval_output.strip("[]").split(",")
        success_list = [int(x.strip()) for x in success_list]
    except Exception:
        raise ValueError(f"Unexpected evaluator output: {eval_output}")

    return success_list

def master_evaluator(response_transcript, word2cat, english_transcript):
    """
    Run lexical acquisition, syntax evaluation, and conversational success checks in parallel.

    Parameters
    ----------
    transcript : list[str]
        Sentences spoken by test_LLM (in conlang).
    word2cat : dict
        Mapping of conlang word → category.
    conlang_vocab : set
        Set of valid conlang words.
    english_transcript : str
        Full conversation translated into English.
    api_key : str
        OpenRouter API key.

    Returns
    -------
    results : dict
        Consolidated results from all three evaluators.
    """

    conlang_vocab = {
        "nuzikebe", "lewu", "lonovo", "mepuroko", "tamipe", "bonava", "kaleno", "mureti", "honiva", "yakuro",
        "velupa", "torime", "zaneko", "sokune", "nirako", "fureta", "gomira", "wateko", "rupone", "deluvo",
        "pimeta", "doraku", "tenova", "silupo", "kanive", "volira", "numeko", "harivo", "jutena", "boreki",
        "zanuro", "nupade", "fuxoga", "jovi", "bexuje", "tono", "zepo", "ruhu", "ziso", "doba", "nafege",
        "zere", "gega", "cufunina", "yemoli", "vusucimu", "fujara", "wibavoyi", "govi", "kisebi", "yeyi",
        "codu", "wulecunu", "nuyixo", "siyi", "muqi", "pobo", "gowi", "cije", "yogekafi", "cesogi",
        "rotifone", "kaqigola", "dalisi", "zaxo", "qomuwo", "pacaqu", "tohafobo", "kejicifa", "rigixi", "xedumo",
        "cufizafa", "hoxixo", "tafinasu", "kofubica", "napahopa", "nepi", "kupe", "vowa", "pala", "cijuwaji",
        "zexono", "sokazato", "depe", "cidiga", "nuri", "mepe", "zucahude", "qinanu", "lavolepa", "qofawo",
        "gazi", "qune", "fute", "lepeqome", "deguwire", "suponidi", "cilesohi", "didekake", "qitube", "zasani",
        "ziraqifu", "pawu", "xajavi", "peya", "rogima", "zabawo", "rohewu", "lakebi", "vubeguho", "rife",
        "givosi", "rogaxi", "reqofuze", "lodaxe", "voleseqa", "lofetoqa", "lavo", "ditawesi", "wiqoto", "kelekoci",
        "qawixa", "dobi", "nenone", "pujuyi", "fafiko", "kapeja", "beru", "fejuvu", "hapecega", "pezaxa",
        "lelekopi", "vegojaja", "neduzifa", "dekayima", "xifugelo", "suwa", "codowosa", "cuha", "lejodeyo", "vamosa",
        "nukenunu", "xicayiza", "gaqa", "dozozo", "budepilu", "vayiwi", "moqo", "cujemoci", "huya", "gewefi",
        "hubame", "soqeyodo", "lohayoqa", "pakiru", "giziji", "tesaqusi", "nakoco", "becoxeho", "lehubepa", "ca",
        "muwaqu", "komixo", "wa", "gorakavi", "noci", "qizohebu", "fode", "safiru", "yoyaboro", "zaco",
        "gosayuye", "qiqihiwu", "tehe", "yuxijepu", "godi", "juqoguge", "canu", "lohemazo", "fekagu", "wamize",
        "zomucu", "fopi", "laxofovi", "suzudo", "cito", "yupo", "vironega", "jejacohi", "juzenuge", "ninopu",
        "fefafo", "naqimi", "zigewe", "baje", "nutinage", "yafevi", "varituja", "sizoda", "nanila", "yepozoba",
        "raxu", "dakozumu", "cumohe", "towuko", "gotido", "hiqaqiha", "vemotemo", "powi", "vikumubu", "decini",
        "vutape", "zocuko", "tawacu", "cofulaxu", "akih", "plak", "sank", "denoc"
    }

    results = {}

    # Define tasks
    def task_lexical():
        return lexical_acquisition(response_transcript, conlang_vocab)

    def task_syntax():
        return syntax_evaluator(response_transcript, word2cat, api_key = "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")

    def task_success():
        return evaluate_conversational_success(english_transcript, api_key = "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")

    tasks = {
        "lexical": task_lexical,
        "syntax": task_syntax,
        "success": task_success
    }

    # Run in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = f"Error: {e}"

    # Reshape results for clarity
    return {
        "lexical": {
            "unique_conlang_count": results["lexical"][0],
            "overall_ratio": results["lexical"][1],
            "avg_change": results["lexical"][2],
        },
        "syntax": {
            "strict_scores": results["syntax"][3],
            "lev_scores": results["syntax"][4],
            "avg_change": results["syntax"][2],
        },
        "success": results["success"]
    }

def run_experiment(models, round_lengths, translator, word2cat, api_key):
    """
    Run the full experiment across multiple models and round configurations.

    Parameters
    ----------
    models : list[str]
        List of model names/identifiers (test LLMs).
    round_lengths : list[int]
        List of round lengths (number of turns per round).
    translator : ConlangTranslator
        Translator instance.
    word2cat : dict
        Mapping of conlang word → category.
    api_key : str
        OpenRouter API key.
    """

    for model_name in models:
        results_file = f"results_{model_name.replace('/', '_').replace(':', '_')}.txt"
        conv_file = f"conversation_{model_name.replace('/', '_').replace(':', '_')}.txt"
        total_rounds = len(round_lengths)

        with open(results_file, "w", encoding="utf-8") as f, open(conv_file, "w", encoding="utf-8") as conv_f:
            f.write(f"=== Results for {model_name} ===\n")

            # Outer progress bar for rounds
            with tqdm(total=total_rounds, desc=f"Model {model_name} rounds", position=0) as pbar_rounds:
                for round_idx, num_turns in enumerate(round_lengths, 1):
                    f.write(f"\n--- Round {round_idx} ({num_turns} turns) ---\n")

                    # Initialize transcripts
                    conv_history = ["ziso akih"]   # starting message
                    response_transcript = []
                    english_transcript = ""
                    conversation_transcript = []  # NEW variable for saving full dialogue

                    # Inner progress bar for turns
                    for _ in tqdm(range(num_turns), desc=f"Round {round_idx}", position=1, leave=False):
                        # Step 1. Get response from test_LLM
                        test_response = test_LLM(api_key, conv_history, model_name)

                        # Step 2. Append test_LLM response to transcripts
                        conv_history.append(test_response)
                        response_transcript.append(test_response)
                        conversation_transcript.append(f"Foreigner: {test_response}")

                        # Step 3. Send to toffi
                        toffi_output = toffi(translator, test_response)

                        # Step 4. Append English outputs to english_transcript
                        english_transcript += f"{toffi_output['cl_to_en']} -> {toffi_output['tribal_english_response']}\n"

                        # Step 5. Append final conlang output to conv_history
                        conv_history.append(toffi_output["final_conlang"])
                        conversation_transcript.append(f"Tribe: {toffi_output}")

                    # Step 6. Evaluate at end of round
                    metrics = master_evaluator(response_transcript, word2cat, english_transcript)

                    # Step 7. Save results
                    f.write(f"Lexical metrics: {metrics['lexical']}\n")
                    f.write(f"Syntax metrics: {metrics['syntax']}\n")
                    f.write(f"Success metrics: {metrics['success']}\n")
                    f.flush()

                    # Step 8. Save conversation transcript
                    conv_f.write(f"\n--- Round {round_idx} ({num_turns} turns) ---\n")
                    conv_f.write("\n".join(conversation_transcript))
                    conv_f.write("\n")
                    conv_f.flush()

                    pbar_rounds.update(1)  # update outer progress bar

        print(f"✅ Finished experiment for {model_name}, results saved to {results_file} and {conv_file}")


def main():
    df = pd.read_csv("conlang_vocab.csv")

    # Build dictionaries
    cl_to_en = dict(zip(df["Conlang"], df["English"]))
    en_to_cl = defaultdict(list)
    for cl_word, en_word in zip(df["Conlang"], df["English"]):
        en_to_cl[en_word].append(cl_word)

    translator = ConlangTranslator(cl_to_en, en_to_cl, api_key="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")

    # Word-to-category mapping
    word2cat = dict(zip(df["Conlang"], df["Category"]))

    # Run experiment
    FAST_MODELS = [
        # "deepseek/deepseek-r1-0528:free",
        # "meta-llama/llama-3.3-8b-instruct:free",
        "openai/gpt-4o-mini",
        # "google/gemini-2.5-flash",
        # "anthropic/claude-3.5-haiku"
        # "openai/gpt-3.5-turbo-0125",
    ]
    ROUND_LENGTHS = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]

    run_experiment(FAST_MODELS, ROUND_LENGTHS, translator, word2cat, api_key="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")


if __name__ == "__main__":
    main()