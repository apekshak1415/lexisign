import re

# ─── HappyTransformer ─────────────────────────────────────────────────────────
try:
    from happytransformer import HappyTextToText, TTSettings
    print("📦 Loading grammar model...")
    _happy = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    _args  = TTSettings(num_beams=5, min_length=1, max_length=100)
    GRAMMAR_AVAILABLE = True
    print("✅ Grammar model loaded!")
except Exception as e:
    print(f"[WARN] HappyTransformer not available: {e}")
    GRAMMAR_AVAILABLE = False

# ─── ISL word normalisation ───────────────────────────────────────────────────
NORM_MAP = {
    "I": "I", "YOU": "you", "HE": "he", "SHE": "she",
    "WE": "we", "THEY": "they", "ME": "I", "MY": "my",
    "HELLO": "Hello", "HI": "Hi", "GOODBYE": "Goodbye",
    "GOOD MORNING": "Good morning", "GOOD AFTERNOON": "Good afternoon",
    "GOOD EVENING": "Good evening",
    "THANK YOU": "Thank you", "THANK": "Thank you",
    "SORRY": "Sorry", "PLEASE": "please", "WELCOME": "You are welcome",
    "YES": "Yes", "NO": "No",
    "WHAT": "what", "WHERE": "where", "WHEN": "when",
    "WHO": "who", "WHY": "why", "HOW": "how",
    "HOW MUCH": "how much", "HOW MANY": "how many",
    "WHICH": "which", "DO YOU": "do you",
    "COME": "come", "GO": "go", "EAT": "eat", "DRINK": "drink",
    "SLEEP": "sleep", "HELP": "help", "WANT": "want", "NEED": "need",
    "KNOW": "know", "UNDERSTAND": "understand", "STOP": "stop", "WAIT": "wait",
    "HAPPY": "happy", "SAD": "sad", "ANGRY": "angry", "TIRED": "tired",
    "SICK": "sick", "HUNGRY": "hungry", "THIRSTY": "thirsty",
    "PAIN": "pain", "FINE": "fine", "SCARED": "scared",
    "HOME": "home", "HOSPITAL": "hospital", "SCHOOL": "school",
    "BATHROOM": "bathroom", "FOOD": "food", "WATER": "water",
    "MEDICINE": "medicine", "MONEY": "money", "TIME": "time", "PHONE": "phone",
    "GOOD": "good", "BAD": "bad", "BIG": "big", "SMALL": "small",
    "FAST": "fast", "SLOW": "slow", "HOT": "hot", "COLD": "cold",
    "MORE": "more", "ENOUGH": "enough",
    "MY NAME": "my name is",
    "HELLO": "Hello",
}

# ─── ISL pattern → English expansion ─────────────────────────────────────────
# ISL drops: is/am/are, a/an/the, auxiliary verbs
# We re-insert them based on common patterns

ISL_PATTERNS = [
    # Question patterns
    (r"^what time now$",          "What is the time now?"),
    (r"^what time\??$",           "What is the time?"),
    (r"^where bathroom\??$",      "Where is the bathroom?"),
    (r"^where hospital\??$",      "Where is the hospital?"),
    (r"^where school\??$",        "Where is the school?"),
    (r"^where home\??$",          "Where is home?"),
    (r"^where (.+)\??$",          r"Where is \1?"),
    (r"^what (.+)\??$",           r"What is \1?"),
    (r"^who (.+)\??$",            r"Who is \1?"),
    (r"^how much (.+)\??$",       r"How much does \1 cost?"),
    (r"^do you (.+)\??$",         r"Do you \1?"),
    # YOU + feeling → Are you feeling?
    (r"^you (hungry|thirsty|tired|sick|fine|happy|sad|angry|scared|okay)\??$",
     r"Are you \1?"),
    # I + feeling → I am feeling
    (r"^i (hungry|thirsty|tired|sick|fine|happy|sad|angry|scared|okay)$",
     r"I am \1."),
    # I + verb → I want/need/...
    (r"^i (want|need|understand|know|come|go|eat|drink|sleep|help)$",
     r"I \1."),
    # NEED + noun
    (r"^(i )?need (.+)$",        r"I need \2."),
    # WANT + noun
    (r"^(i )?want (.+)$",        r"I want \2."),
    # HELP + me
    (r"^help me$",               "Please help me."),
    (r"^need help$",             "I need help."),
    # MY NAME
    (r"^my name is (.+)$",       r"My name is \1."),
]

def _normalise(words):
    """Convert signed word list to rough English using NORM_MAP."""
    out   = []
    i     = 0
    upper = [w.upper() for w in words]
    while i < len(upper):
        matched = False
        for length in range(min(3, len(upper)-i), 0, -1):
            phrase = " ".join(upper[i:i+length])
            if phrase in NORM_MAP:
                out.append(NORM_MAP[phrase])
                i += length
                matched = True
                break
        if not matched:
            out.append(words[i].lower())
            i += 1
    return out

def _apply_isl_patterns(text):
    """Try to match ISL topic-comment patterns and expand to full English."""
    t = text.strip().lower().rstrip("?.")
    for pattern, replacement in ISL_PATTERNS:
        m = re.match(pattern, t, re.IGNORECASE)
        if m:
            result = m.expand(replacement) if r"\\" in repr(replacement) or "\\1" in replacement or "\\2" in replacement else replacement
            # Capitalise first letter
            return result[0].upper() + result[1:]
    return None

def correct_sentence(words):
    """
    Main entry point called by realtime_translator and app.py.
    1. Normalise ISL words → rough English
    2. Try ISL pattern matching for common structures
    3. Pass through HappyTransformer for grammar polish
    """
    if not words:
        return ""

    normalised = _normalise(words)
    rough      = " ".join(normalised)
    print(f"  [NLP] Input words : {words}")
    print(f"  [NLP] Normalised  : {rough}")

    # Try pattern match first
    pattern_result = _apply_isl_patterns(rough)
    if pattern_result:
        print(f"  [NLP] Pattern hit : {pattern_result}")
        return pattern_result

    # Fall through to ML grammar correction
    if GRAMMAR_AVAILABLE:
        try:
            result    = _happy.generate_text(f"grammar: {rough}", args=_args)
            corrected = result.text.strip()
            print(f"  [NLP] ML corrected: {corrected}")
        except Exception as e:
            print(f"  [NLP] ML failed: {e}")
            corrected = rough
    else:
        corrected = rough

    # Capitalise + punctuate
    if corrected:
        corrected = corrected[0].upper() + corrected[1:]
        if not corrected.endswith((".", "!", "?")):
            corrected += "."

    return corrected


if __name__ == "__main__":
    tests = [
        ["WHAT", "TIME", "NOW"],
        ["WHERE", "BATHROOM"],
        ["YOU", "HUNGRY"],
        ["I", "TIRED"],
        ["NEED", "WATER"],
        ["WHERE", "HOSPITAL"],
        ["HELP"],
        ["THANK YOU"],
        ["WHAT", "YOUR", "NAME"],
        ["I", "PAIN"],
        ["WHERE", "HOME"],
    ]
    print("\n── ISL → English NLP Test ────────────────────")
    for w in tests:
        print(f"\n  Input : {w}")
        print(f"  Output: {correct_sentence(w)}")
    print("──────────────────────────────────────────────")