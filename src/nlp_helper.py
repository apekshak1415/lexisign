from openai import OpenAI
import re

# Set your key here - leave empty string to use offline mode
OPENAI_API_KEY = "YOUR_KEY_HERE"

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Rule-based sentence correction (offline fallback)
WORD_FIXES = {
    "I_ME_MINE_MY": "I",
    "HELLO_HI": "Hello",
    "LIKE_LOVE": "love",
    "COLLEGE_SCHOOL": "school",
    "OLD_AGE": "old",
    "DON'T CARE": "don't care",
    "ON THE WAY": "on the way",
    "MEAN IT": "mean it",
    "TAKE CARE": "take care",
    "TAKE TIME": "take time",
    "TURN ON": "turn on",
    "SO MUCH": "so much",
    "SOME HOW": "somehow",
    "SOME ONE": "someone",
}

# Simple grammar rules
SENTENCE_STARTERS = ["I", "YOU", "WE", "HELLO", "PLEASE", "SORRY", "THANK"]

def rule_based_correction(words: list) -> str:
    """Offline rule-based sentence builder."""
    if not words:
        return ""

    # Normalize words
    corrected = []
    for word in words:
        w = word.upper().strip()
        if w in WORD_FIXES:
            corrected.append(WORD_FIXES[w])
        else:
            corrected.append(w.capitalize())

    sentence = " ".join(corrected)

    # Capitalize first letter, add period
    sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
    if sentence and not sentence.endswith((".", "!", "?")):
        sentence += "."

    return sentence

def correct_sentence(words: list) -> str:
    """
    Convert detected sign words to a proper sentence.
    Uses ChatGPT if available, falls back to rule-based offline.
    """
    if not words:
        return ""

    raw_text = " ".join(words)

    # Try ChatGPT first
    if client and OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_KEY_HERE":
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that converts raw sign language words "
                            "into proper grammatical English sentences. "
                            "The input is a sequence of words detected from Indian Sign Language. "
                            "Return ONLY the corrected sentence, nothing else. "
                            "Keep the meaning as close to the original words as possible."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Convert these ISL words into a proper sentence: {raw_text}"
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  ChatGPT unavailable ({e}), using offline NLP...")

    # Offline fallback
    return rule_based_correction(words)

def test_nlp():
    test_cases = [
        ["HELLO_HI", "I_ME_MINE_MY", "HUNGRY", "WANT", "FOOD"],
        ["YOU", "HELP", "ME", "PLEASE"],
        ["I_ME_MINE_MY", "TIRED", "WANT", "SLEEP"],
        ["THANK", "YOU", "FRIEND"],
        ["SORRY", "I_ME_MINE_MY", "LATE"],
    ]
    print("🧪 Testing NLP Helper...\n")
    for words in test_cases:
        result = correct_sentence(words)
        print(f"Input:  {' '.join(words)}")
        print(f"Output: {result}")
        print()

if __name__ == "__main__":
    test_nlp()