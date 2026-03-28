"""
AQLES Quality Scorer — Gradio Demo with 3D Visualisation
==========================================================
Run locally:  python app.py
On Colab:     paste this code in a cell, it auto-launches with share=True
"""

import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# ── Quality lexicon (200 words x 5 tiers) ────────────────────
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

TIER_NAMES = {4: "Exceptional", 3: "Excellent", 2: "Good",
              1: "Mediocre", 0: "Terrible"}
TIER_COLORS = {0: "#F44336", 1: "#FF9800", 2: "#FFC107",
               3: "#4CAF50", 4: "#2196F3"}

MODEL_MAP = {
    "DistilBERT": "distilbert-base-uncased",
    "BERT-base": "bert-base-uncased",
    "GPT-2": "gpt2",
}


def score_to_tier(s):
    if s >= 0.90: return 4
    if s >= 0.78: return 3
    if s >= 0.45: return 2
    if s >= 0.15: return 1
    return 0


MODEL_CACHE = {}
PROBE_CACHE = {}
HIDDEN_CACHE = {}


def load_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True
    ).eval()
    MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def extract_hidden(sentences, model_name, batch_size=32):
    tokenizer, model = load_model(model_name)
    is_decoder = model_name == "gpt2"
    all_layers = None
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=64)
        with torch.no_grad():
            out = model(**enc)
        for li, hs in enumerate(out.hidden_states):
            if is_decoder:
                lengths = enc["attention_mask"].sum(dim=1) - 1
                vecs = hs[torch.arange(hs.size(0)), lengths].numpy()
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vecs = ((hs * mask).sum(1) / mask.sum(1)).numpy()
            if all_layers is None:
                all_layers = {l: [] for l in range(len(out.hidden_states))}
            all_layers[li].append(vecs)
    return {l: np.vstack(c) for l, c in all_layers.items()}


def build_data():
    words, scores, tiers, sentences, wids = [], [], [], [], []
    for wid, (w, s) in enumerate(QUALITY_LEXICON.items()):
        for tmpl in TEMPLATES:
            words.append(w)
            scores.append(s)
            tiers.append(score_to_tier(s))
            sentences.append(tmpl.format(word=w))
            wids.append(wid)
    return (words, np.array(scores), np.array(tiers),
            sentences, np.array(wids))


def get_probes(model_name):
    if model_name in PROBE_CACHE:
        return PROBE_CACHE[model_name], HIDDEN_CACHE[model_name]
    words, scores, tiers, sentences, wids = build_data()
    hidden = extract_hidden(sentences, model_name)
    HIDDEN_CACHE[model_name] = (hidden, np.array(tiers), words)
    probes = {}
    for li, X in hidden.items():
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        gkf = GroupKFold(n_splits=5)
        best_a, best_r2 = 1.0, -np.inf
        for a in [0.01, 0.1, 1, 10, 100, 500, 1000, 2000, 5000]:
            fr2 = []
            for tr, te in gkf.split(Xs, scores, wids):
                r = Ridge(alpha=a).fit(Xs[tr], scores[tr])
                p = r.predict(Xs[te])
                ss_r = np.sum((scores[te] - p) ** 2)
                ss_t = np.sum((scores[te] - scores[te].mean()) ** 2)
                fr2.append(1 - ss_r / ss_t)
            mr2 = np.mean(fr2)
            if mr2 > best_r2:
                best_r2, best_a = mr2, a
        ridge = Ridge(alpha=best_a).fit(Xs, scores)
        probes[li] = {"scaler": sc, "ridge": ridge,
                      "cv_r2": best_r2, "alpha": best_a}
    PROBE_CACHE[model_name] = probes
    return probes, (hidden, np.array(tiers), words)


# ── Tab 1: Score ──────────────────────────────────────────────
def score_word(word, model_choice):
    word = word.strip().lower()
    if not word:
        return "Please enter a word."
    model_name = MODEL_MAP[model_choice]
    probes, _ = get_probes(model_name)
    sentences = [tmpl.format(word=word) for tmpl in TEMPLATES]
    hidden = extract_hidden(sentences, model_name)
    results = []
    for li in sorted(hidden.keys()):
        Xs = probes[li]["scaler"].transform(hidden[li])
        preds = probes[li]["ridge"].predict(Xs)
        mean_pred = float(np.mean(preds))
        tier = score_to_tier(mean_pred)
        results.append({
            "layer": li, "score": round(mean_pred, 3),
            "tier": TIER_NAMES[tier],
            "cv_r2": round(probes[li]["cv_r2"], 3),
        })
    best = max(results, key=lambda r: r["cv_r2"])
    known = word in QUALITY_LEXICON
    true_score = QUALITY_LEXICON.get(word, None)
    out = f"## {word.upper()}\n\n"
    if known:
        out += f"**Known word** — True score: {true_score:.2f} ({TIER_NAMES[score_to_tier(true_score)]})\n\n"
    else:
        out += "**Unknown word** — not in the 200-word lexicon (out-of-vocabulary prediction)\n\n"
    out += f"**Model:** {model_choice} | **Best layer:** L{best['layer']} (CV R² = {best['cv_r2']})\n\n"
    out += f"**Predicted quality:** {best['score']:.3f} → **{best['tier']}**\n\n"
    out += "| Layer | Score | Tier | R² |\n|---|---|---|---|\n"
    for r in results:
        out += f"| L{r['layer']} | {r['score']:.3f} | {r['tier']} | {r['cv_r2']} |\n"
    return out


# ── Tab 2: 3D ─────────────────────────────────────────────────
def make_3d_plot(model_choice, layer_idx):
    model_name = MODEL_MAP[model_choice]
    probes, (hidden, tiers, words) = get_probes(model_name)
    max_layer = max(hidden.keys())
    layer_idx = max(0, min(int(layer_idx), max_layer))
    X = hidden[layer_idx]
    pca = PCA(n_components=3)
    X3d = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100
    fig = go.Figure()
    for t in [0, 1, 2, 3, 4]:
        mask = tiers == t
        if not np.any(mask):
            continue
        hover_texts = [words[i] for i in range(len(words)) if mask[i]]
        fig.add_trace(go.Scatter3d(
            x=X3d[mask, 0], y=X3d[mask, 1], z=X3d[mask, 2],
            mode="markers",
            marker=dict(size=3, color=TIER_COLORS[t], opacity=0.6),
            name=f"{TIER_NAMES[t]} (T{t})",
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))
    r2 = probes[layer_idx]["cv_r2"]
    fig.update_layout(
        title=f"{model_choice} Layer {layer_idx} (R²={r2:.3f})",
        scene=dict(
            xaxis_title=f"PC1 ({var_exp[0]:.1f}%)",
            yaxis_title=f"PC2 ({var_exp[1]:.1f}%)",
            zaxis_title=f"PC3 ({var_exp[2]:.1f}%)",
        ),
        width=700, height=550,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98, font=dict(size=11)),
    )
    return fig


# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(title="AQLES", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # AQLES — Quality Scorer & 3D Geometry Explorer
    How do transformers encode the difference between *terrible* and *masterful*?
    Type a word and find out. Then explore the 3D geometry.

    **First run trains probes (~2 min per model). After that, instant.**
    """)

    with gr.Tabs():
        with gr.Tab("🎯 Score a Word"):
            with gr.Row():
                word_input = gr.Textbox(
                    label="Enter a quality word",
                    placeholder="excellent, mediocre, irreproachable...",
                    scale=3,
                )
                model_dd = gr.Dropdown(
                    choices=["DistilBERT", "BERT-base", "GPT-2"],
                    value="DistilBERT", label="Model", scale=1,
                )
            score_btn = gr.Button("Score", variant="primary")
            score_out = gr.Markdown()
            score_btn.click(score_word, [word_input, model_dd], score_out)
            gr.Examples(
                examples=[
                    ["excellent", "DistilBERT"],
                    ["terrible", "GPT-2"],
                    ["irreproachable", "BERT-base"],
                    ["mediocre", "GPT-2"],
                    ["catastrophic", "GPT-2"],
                ],
                inputs=[word_input, model_dd],
            )

        with gr.Tab("🌐 3D Quality Geometry"):
            gr.Markdown("""
            **Interactive 3D PCA** of all 2,000 probing sentences at a chosen layer.
            Each dot = one sentence, coloured by quality tier.
            **Drag** to rotate, **scroll** to zoom, **hover** to see the word.

            Compare Layer 0 (raw embedding) with the best layer to see tiers separate.

            DistilBERT has layers 0–6. BERT-base and GPT-2 have layers 0–12.
            """)
            with gr.Row():
                viz_model = gr.Dropdown(
                    choices=["DistilBERT", "BERT-base", "GPT-2"],
                    value="DistilBERT", label="Model", scale=1,
                )
                viz_layer = gr.Number(
                    value=5, label="Layer",
                    precision=0, minimum=0, maximum=12,
                    scale=1,
                )
            viz_btn = gr.Button("Generate 3D Plot", variant="primary")
            viz_plot = gr.Plot(label="3D Quality Geometry")
            viz_btn.click(make_3d_plot, [viz_model, viz_layer], viz_plot)

    gr.Markdown("""
    ---
    *200 words × 10 templates = 2,000 sentences. Ridge probes on frozen hidden states.
    GroupKFold CV, zero data leakage.* ·
    Apache 2.0 · Fabrice Fils-Aimé ·
    [GitHub](https://github.com/fabthebest/aqles)
    """)

if __name__ == "__main__":
    demo.launch(share=True)
