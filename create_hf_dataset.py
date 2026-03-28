"""
Generate the AQLES quality lexicon dataset and push to HuggingFace Hub.

Usage:
    pip install datasets huggingface_hub
    huggingface-cli login
    python create_hf_dataset.py
"""

import pandas as pd
from datasets import Dataset

# ── Same lexicon as in the notebook ──────────────────────────
QUALITY_LEXICON = {
    "exceptional":1.00, "extraordinary":1.00, "magnificent":1.00,
    "breathtaking":0.98, "transcendent":0.98, "masterful":0.97,
    "sublime":0.97, "flawless":0.96, "faultless":0.96,
    "groundbreaking":0.96, "impeccable":0.95, "supreme":0.95,
    "incomparable":0.95, "unparalleled":0.95, "phenomenal":0.94,
    "superlative":0.94, "nonpareil":0.94, "matchless":0.94,
    "prodigious":0.93, "virtuosic":0.93, "peerless":0.93,
    "consummate":0.93, "ineffable":0.93, "paramount":0.93,
    "irreproachable":0.92, "stupendous":0.92, "majestic":0.92,
    "immaculate":0.92, "quintessential":0.92, "exemplary":0.91,
    "towering":0.91, "sterling":0.91, "august":0.91,
    "definitive":0.91, "transcendental":0.91, "luminous":0.90,
    "resplendent":0.90, "pristine":0.90, "supernal":0.90,
    "celestial":0.90,
    "beautiful":0.88, "excellent":0.87, "outstanding":0.87,
    "stellar":0.87, "first-rate":0.87, "superb":0.86,
    "brilliant":0.86, "elite":0.86, "top-notch":0.86,
    "first-class":0.86, "remarkable":0.85, "prestigious":0.85,
    "high-quality":0.85, "invaluable":0.85, "impressive":0.84,
    "accomplished":0.84, "inspired":0.84, "priceless":0.84,
    "superior":0.84, "splendid":0.83, "wonderful":0.83,
    "noteworthy":0.83, "polished":0.83, "esteemed":0.83,
    "distinguished":0.85, "premier":0.83, "admirable":0.83,
    "fantastic":0.82, "commendable":0.82, "praiseworthy":0.82,
    "proficient":0.82, "prime":0.82, "refined":0.82,
    "formidable":0.85, "laudable":0.81, "meritorious":0.81,
    "reputable":0.81, "laudatory":0.81, "creditable":0.80,
    "venerable":0.80,
    "great":0.70, "good":0.68, "solid":0.65,
    "efficient":0.65, "reliable":0.65, "effective":0.67,
    "competent":0.63, "capable":0.62, "practical":0.62,
    "consistent":0.63, "dependable":0.63, "stable":0.60,
    "decent":0.60, "viable":0.60, "respectable":0.60,
    "useful":0.63, "sound":0.62, "appropriate":0.58,
    "sufficient":0.57, "functional":0.55, "adequate":0.55,
    "standard":0.55, "modest":0.55, "presentable":0.55,
    "satisfactory":0.52, "serviceable":0.53, "workable":0.54,
    "acceptable":0.50, "reasonable":0.50, "conventional":0.50,
    "average":0.50, "passable":0.50, "moderate":0.50,
    "fair":0.48, "middling":0.48, "unremarkable":0.48,
    "routine":0.48, "prosaic":0.46, "ordinary":0.45,
    "tolerable":0.47,
    "pedestrian":0.30, "mediocre":0.30, "rudimentary":0.28,
    "derivative":0.28, "subpar":0.28, "superficial":0.28,
    "incomplete":0.28, "redundant":0.28, "cursory":0.25,
    "disappointing":0.25, "underwhelming":0.25, "lackluster":0.25,
    "inconsistent":0.25, "forgettable":0.25, "unimpressive":0.25,
    "convoluted":0.25, "tedious":0.25, "banal":0.25,
    "shallow":0.25, "flawed":0.25, "muddled":0.22,
    "amateurish":0.22, "inferior":0.22, "unreliable":0.22,
    "substandard":0.22, "uninspired":0.22, "problematic":0.22,
    "clunky":0.20, "deficient":0.20, "flimsy":0.20,
    "faulty":0.20, "incoherent":0.20, "unsatisfactory":0.20,
    "dull":0.22, "defective":0.18, "shoddy":0.18,
    "poor":0.18, "lacking":0.18, "weak":0.15,
    "inadequate":0.15,
    "terrible":0.05, "awful":0.05, "miserable":0.05,
    "unacceptable":0.05, "horrible":0.04, "dreadful":0.04,
    "shameful":0.04, "wretched":0.04, "pitiful":0.04,
    "lamentable":0.04, "intolerable":0.04, "unbearable":0.04,
    "grotesque":0.04, "appalling":0.03, "deplorable":0.03,
    "despicable":0.03, "abysmal":0.03, "atrocious":0.03,
    "revolting":0.03, "egregious":0.03, "inexcusable":0.03,
    "insufferable":0.03, "hideous":0.03, "reprehensible":0.02,
    "catastrophic":0.02, "disastrous":0.02, "contemptible":0.02,
    "detestable":0.02, "execrable":0.02, "shameless":0.02,
    "heinous":0.02, "indefensible":0.02, "unconscionable":0.02,
    "repugnant":0.02, "odious":0.02, "abhorrent":0.02,
    "vile":0.02, "loathsome":0.02, "worthless":0.01,
    "abominable":0.01,
}

TEMPLATES = [
    "The overall quality of this work is {word}.",
    "This piece of work is truly {word}.",
    "The performance was {word} in every respect.",
    "I would describe this result as {word}.",
    "From a scientific standpoint, this contribution is {word}.",
    "Reviewers unanimously agreed the submission was {word}.",
    "The committee rated this project as {word}.",
    "Colleagues described the output as {word} across the board.",
    "After careful review, the quality was deemed {word}.",
    "The final evaluation concluded that this work is {word}.",
]


def score_to_tier(s):
    if s >= 0.90: return 4
    if s >= 0.78: return 3
    if s >= 0.45: return 2
    if s >= 0.15: return 1
    return 0


TIER_NAMES = {4: "Exceptional", 3: "Excellent", 2: "Good",
              1: "Mediocre", 0: "Terrible"}


def build_dataset():
    rows = []
    for wid, (word, score) in enumerate(QUALITY_LEXICON.items()):
        tier = score_to_tier(score)
        for tid, tmpl in enumerate(TEMPLATES):
            rows.append({
                "word": word,
                "word_id": wid,
                "quality_score": score,
                "tier": tier,
                "tier_name": TIER_NAMES[tier],
                "template_id": tid,
                "sentence": tmpl.format(word=word),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    print(f"Dataset: {len(df)} rows, {df.word.nunique()} words, "
          f"{df.tier.nunique()} tiers")
    print(df.groupby("tier_name")["word"].nunique())

    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(
        "fabthebest/aqles-quality-lexicon",
        private=False,
    )
    print("Pushed to HuggingFace Hub.")
