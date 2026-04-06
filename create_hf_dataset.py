"""
Create and push the AQLES multilingual quality lexicon dataset to HuggingFace Hub.

Covers:
  - English  : 200 words, 5 tiers, validated against NRC VAD
  - French   : ~100 words, 5 tiers, unvalidated (author construction, V6)
  - Spanish  : ~100 words, 5 tiers, unvalidated (author construction, V6)

Usage:
    pip install datasets huggingface_hub pandas
    huggingface-cli login
    python create_hf_dataset.py
"""

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel

# ── English lexicon (200 words, V1) ──────────────────────────
LEXICON_EN = {
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

# ── French lexicon (~100 words, V6, unvalidated) ──────────────
# WARNING: constructed by a non-native speaker, not reviewed by a linguist.
# Scores calibrated by analogy with English NRC VAD.
# See technical report Section 5 (Limitations) before using for research.
LEXICON_FR = {
    # T4 Exceptional (score >= 0.90)
    "exceptionnel":0.95, "magistral":0.94, "sublime":0.93,
    "irréprochable":0.92, "parfait":0.91, "immaculé":0.91,
    "inégalable":0.92, "prodigieux":0.93, "incomparable":0.92,
    "transcendant":0.91, "lumineux":0.90, "resplendissant":0.90,
    "souverain":0.91, "supérieur":0.90, "fabuleux":0.90,
    "vertueux":0.91, "insurpassable":0.93, "quintessentiel":0.92,
    "exemplaire":0.91, "définitif":0.90,
    # T3 Excellent (0.78 to 0.89)
    "excellent":0.87, "remarquable":0.85, "brillant":0.86,
    "distingué":0.85, "admirable":0.83, "impressionnant":0.84,
    "louable":0.81, "estimé":0.83, "prestigieux":0.85,
    "accompli":0.84, "commendable":0.82, "émérite":0.83,
    "raffiné":0.82, "splendide":0.83, "formidable":0.85,
    "accompli":0.84, "méritoire":0.81, "honorable":0.80,
    "supérieur":0.84, "de premier ordre":0.86,
    # T2 Good (0.45 to 0.77)
    "bon":0.68, "solide":0.65, "efficace":0.67,
    "compétent":0.63, "fiable":0.65, "adéquat":0.55,
    "satisfaisant":0.52, "convenable":0.60, "correct":0.60,
    "fonctionnel":0.55, "raisonnable":0.50, "acceptable":0.50,
    "standard":0.55, "ordinaire":0.48, "passable":0.50,
    "modeste":0.55, "praticable":0.54, "stable":0.60,
    "suffisant":0.57, "approprié":0.58,
    # T1 Mediocre (0.15 to 0.44)
    "médiocre":0.30, "insuffisant":0.22, "décevant":0.25,
    "terne":0.25, "banal":0.25, "superficiel":0.28,
    "rudimentaire":0.28, "lacunaire":0.25, "inégal":0.25,
    "oubliable":0.25, "peu inspiré":0.22, "faible":0.20,
    "déficient":0.20, "maladroit":0.22, "peu convaincant":0.25,
    "bâclé":0.20, "incomplet":0.28, "confus":0.22,
    "peu rigoureux":0.25, "sous-standard":0.22,
    # T0 Terrible (below 0.15)
    "abominable":0.03, "catastrophique":0.02, "désastreux":0.03,
    "épouvantable":0.04, "atroce":0.03, "lamentable":0.04,
    "déplorable":0.03, "exécrable":0.02, "répréhensible":0.02,
    "inadmissible":0.05, "inacceptable":0.05, "horrible":0.04,
    "affreux":0.04, "pitoyable":0.04, "consternant":0.04,
    "honteux":0.04, "indigne":0.03, "nul":0.05,
    "médiocre":0.05, "intolérable":0.04,
}

# ── Spanish lexicon (~100 words, V6, unvalidated) ─────────────
# WARNING: constructed by a non-native speaker, not reviewed by a linguist.
# The 3.3x H5 ratio observed in V6 may reflect word selection bias.
# See technical report Section 5 (Limitations) before using for research.
LEXICON_ES = {
    # T4 Exceptional (score >= 0.90)
    "excepcional":0.95, "magistral":0.94, "sublime":0.93,
    "irreprochable":0.92, "perfecto":0.91, "inmaculado":0.91,
    "incomparable":0.92, "prodigioso":0.93, "extraordinario":0.95,
    "trascendental":0.91, "luminoso":0.90, "resplandeciente":0.90,
    "insuperable":0.93, "quintaesencial":0.92, "ejemplar":0.91,
    "definitivo":0.90, "supremo":0.91, "consagrado":0.90,
    "virtuoso":0.91, "sin par":0.92,
    # T3 Excellent (0.78 to 0.89)
    "excelente":0.87, "notable":0.85, "brillante":0.86,
    "distinguido":0.85, "admirable":0.83, "impresionante":0.84,
    "loable":0.81, "prestigioso":0.85, "sobresaliente":0.85,
    "logrado":0.84, "meritorio":0.81, "refinado":0.82,
    "espléndido":0.83, "formidable":0.85, "encomiable":0.82,
    "emérito":0.83, "honorable":0.80, "elogiable":0.81,
    "superior":0.84, "de primera":0.86,
    # T2 Good (0.45 to 0.77)
    "bueno":0.68, "sólido":0.65, "eficaz":0.67,
    "competente":0.63, "fiable":0.65, "adecuado":0.55,
    "satisfactorio":0.52, "conveniente":0.60, "correcto":0.60,
    "funcional":0.55, "razonable":0.50, "aceptable":0.50,
    "estándar":0.55, "pasable":0.50, "modesto":0.55,
    "suficiente":0.57, "apropiado":0.58, "estable":0.60,
    "viable":0.60, "practicable":0.54,
    # T1 Mediocre (0.15 to 0.44)
    "mediocre":0.30, "insuficiente":0.22, "decepcionante":0.25,
    "anodino":0.25, "superficial":0.28, "rudimentario":0.28,
    "irregular":0.25, "olvidable":0.25, "poco inspirado":0.22,
    "débil":0.20, "deficiente":0.20, "torpe":0.22,
    "incompleto":0.28, "confuso":0.22, "descuidado":0.20,
    "subestándar":0.22, "poco riguroso":0.25, "insulso":0.25,
    "flojo":0.20, "chapucero":0.20,
    # T0 Terrible (below 0.15)
    "abominable":0.03, "catastrófico":0.02, "desastroso":0.03,
    "espantoso":0.04, "atroz":0.03, "lamentable":0.04,
    "deplorable":0.03, "execrable":0.02, "reprehensible":0.02,
    "inadmisible":0.05, "inaceptable":0.05, "horrible":0.04,
    "horrendo":0.04, "penoso":0.04, "bochornoso":0.04,
    "vergonzoso":0.04, "indigno":0.03, "pésimo":0.05,
    "nefasto":0.03, "intolerable":0.04,
}

TEMPLATES_EN = [
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

TEMPLATES_FR = [
    "La qualite generale de ce travail est {word}.",
    "Ce travail est veritablement {word}.",
    "La performance etait {word} a tous egards.",
    "Je decrirais ce resultat comme {word}.",
    "D'un point de vue scientifique, cette contribution est {word}.",
    "Les examinateurs ont unanimement juge la soumission {word}.",
    "Le comite a evalue ce projet comme {word}.",
    "Les collegues ont decrit le resultat comme {word} dans l'ensemble.",
    "Apres un examen attentif, la qualite a ete jugee {word}.",
    "L'evaluation finale a conclu que ce travail est {word}.",
]

TEMPLATES_ES = [
    "La calidad general de este trabajo es {word}.",
    "Este trabajo es verdaderamente {word}.",
    "El rendimiento fue {word} en todos los aspectos.",
    "Describeria este resultado como {word}.",
    "Desde un punto de vista cientifico, esta contribucion es {word}.",
    "Los revisores acordaron unanimemente que la presentacion era {word}.",
    "El comite califico este proyecto como {word}.",
    "Los colegas describieron el resultado como {word} en general.",
    "Tras una revision cuidadosa, la calidad fue considerada {word}.",
    "La evaluacion final concluyo que este trabajo es {word}.",
]

TIER_NAMES = {
    4: "Exceptional",
    3: "Excellent",
    2: "Good",
    1: "Mediocre",
    0: "Terrible",
}

LANGUAGE_CONFIG = {
    "en": {
        "lexicon":   LEXICON_EN,
        "templates": TEMPLATES_EN,
        "validated": True,
        "n_words":   200,
    },
    "fr": {
        "lexicon":   LEXICON_FR,
        "templates": TEMPLATES_FR,
        "validated": False,
        "n_words":   100,
    },
    "es": {
        "lexicon":   LEXICON_ES,
        "templates": TEMPLATES_ES,
        "validated": False,
        "n_words":   100,
    },
}


def score_to_tier(s):
    if s >= 0.90: return 4
    if s >= 0.78: return 3
    if s >= 0.45: return 2
    if s >= 0.15: return 1
    return 0


def build_language_rows(lang_code, lexicon, templates, validated):
    rows = []
    for wid, (word, score) in enumerate(lexicon.items()):
        tier = score_to_tier(score)
        for tid, tmpl in enumerate(templates):
            rows.append({
                "language":          lang_code,
                "word":              word,
                "word_id":           wid,
                "quality_score":     round(float(score), 4),
                "tier":              tier,
                "tier_name":         TIER_NAMES[tier],
                "template_id":       tid,
                "sentence":          tmpl.format(word=word),
                "lexicon_validated": validated,
            })
    return rows


def build_full_dataset():
    all_rows = []
    for lang_code, cfg in LANGUAGE_CONFIG.items():
        rows = build_language_rows(
            lang_code,
            cfg["lexicon"],
            cfg["templates"],
            cfg["validated"],
        )
        all_rows.extend(rows)
        print(f"{lang_code.upper()}: {len(cfg['lexicon'])} words, "
              f"{len(rows)} rows, validated={cfg['validated']}")
    return pd.DataFrame(all_rows)


def print_summary(df):
    print(f"\nTotal rows: {len(df)}")
    print(f"Languages: {df['language'].unique().tolist()}")
    print("\nWords per tier per language:")
    summary = df.groupby(["language", "tier_name"])["word"].nunique().unstack()
    print(summary.to_string())
    print("\nValidation status:")
    print(df.groupby("language")["lexicon_validated"].first().to_string())


if __name__ == "__main__":
    df = build_full_dataset()
    print_summary(df)

    # Split into per-language datasets for cleaner HF navigation
    dataset_dict = {}
    for lang in ["en", "fr", "es"]:
        subset = df[df["language"] == lang].reset_index(drop=True)
        dataset_dict[lang] = Dataset.from_pandas(
            subset, preserve_index=False
        )

    # Also create a combined split
    dataset_dict["all"] = Dataset.from_pandas(
        df.reset_index(drop=True), preserve_index=False
    )

    combined = DatasetDict(dataset_dict)
    print(f"\nDatasetDict splits: {list(combined.keys())}")

    combined.push_to_hub(
        "fabthebest/aqles-quality-lexicon",
        private=False,
        commit_message=(
            "Add French and Spanish lexicons (V6, unvalidated). "
            "English lexicon unchanged (200 words, NRC VAD calibrated)."
        ),
    )
    print("\nPushed to HuggingFace Hub.")
    print("URL: https://huggingface.co/datasets/fabthebest/aqles-quality-lexicon")
