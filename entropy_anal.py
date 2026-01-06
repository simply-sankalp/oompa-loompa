import re
import random
import glob
import numpy as np
from collections import Counter
from scipy.stats import entropy
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import os

os.makedirs("plots/mean_entropy", exist_ok=True)
os.makedirs("plots/clusters", exist_ok=True)

# ----------------------------
# 1. Syllable segmentation
# ----------------------------

VOWELS = set("aeiou")

def cv_segment(token):
    token = token.lower()
    syllables = []
    i = 0

    while i < len(token):
        onset = ""
        nucleus = ""
        coda = ""

        while i < len(token) and token[i] not in VOWELS and len(onset) < 2:
            onset += token[i]
            i += 1

        if i < len(token) and token[i] in VOWELS:
            nucleus = token[i]
            i += 1
        else:
            break

        if i < len(token) and token[i] not in VOWELS:
            coda = token[i]
            i += 1

        syllables.append(onset + nucleus + coda)

    return syllables


def extract_syllables(text):
    syllables = []
    for tok in text.strip().split():
        syllables.extend(cv_segment(tok))
    return syllables


def is_canonical_cv(syllable):
    return sum(c in VOWELS for c in syllable) == 1


# ----------------------------
# 2. Feature computation
# ----------------------------

def canonical_syllable_ratio(syllables):
    if not syllables:
        return 0.0
    return sum(is_canonical_cv(s) for s in syllables) / len(syllables)


def syllable_entropy(syllables):
    if not syllables:
        return 0.0
    counts = Counter(syllables)
    probs = np.array(list(counts.values())) / len(syllables)
    return entropy(probs, base=2)


def conditional_entropy(curr, prev):
    if not curr or not prev:
        return 0.0

    overlap = set(curr) & set(prev)
    p = len(overlap) / len(set(curr))
    p = max(min(p, 0.999), 1e-6)

    return entropy([p, 1 - p], base=2)


# ----------------------------
# 3. Transcript parsing
# ----------------------------

def load_agent_turns(path, agent_prefix):
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(agent_prefix):
                text = line.split(":", 1)[1]
                text = re.sub(r"\(.*?\)", "", text)
                turns.append(text.strip())
    return turns


# ----------------------------
# 4. Feature extraction
# ----------------------------

def extract_features(turns):
    features = []
    all_syllables = []
    prev = None

    for turn in turns:
        syll = extract_syllables(turn)
        all_syllables.extend(syll)

        features.append([
            canonical_syllable_ratio(syll),
            syllable_entropy(syll),
            conditional_entropy(syll, prev),
            len(syll)
        ])

        prev = syll

    return np.array(features), all_syllables


# ----------------------------
# 5. Reduplicative baseline
# ----------------------------

def extract_canonical_inventory(all_syllables):
    return list({s for s in all_syllables if is_canonical_cv(s)})


def generate_reduplicative_utterance(length, inventory):
    s = random.choice(inventory)
    return [s] * length


def reduplicative_baseline_entropy(lengths, inventory, n_samples=1000):
    entropies = []
    for _ in range(n_samples):
        L = int(random.choice(lengths))
        syll = generate_reduplicative_utterance(L, inventory)
        entropies.append(syllable_entropy(syll))
    return np.array(entropies)


# ----------------------------
# 6. Clustering
# ----------------------------

def cluster_turns(X, k=2):
    Xs = StandardScaler().fit_transform(X)
    return GaussianMixture(k, random_state=42).fit_predict(Xs)


# ----------------------------
# 7. Plotting
# ----------------------------

def plot_mean_entropy(mean_e, std_e, baseline_mean, out_path):
    x = np.arange(len(mean_e))

    plt.figure(figsize=(8, 4))
    plt.plot(x, mean_e, label="Observed entropy")
    plt.fill_between(
        x,
        mean_e - std_e,
        mean_e + std_e,
        alpha=0.3,
        label="±1 SD"
    )

    # Reduplicative baseline
    plt.axhline(
        baseline_mean,
        linestyle="--",
        linewidth=2,
        label="Reduplicative baseline"
    )

    plt.xlabel("Turn index")
    plt.ylabel("Syllable entropy")
    plt.title("Mean syllable entropy over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_entropy_clusters(X, labels, out_path):
    plt.figure(figsize=(6, 5))
    for c in sorted(set(labels)):
        idx = labels == c
        plt.scatter(X[idx, 1], X[idx, 2], label=f"Cluster {c}", alpha=0.7)

    plt.xlabel("Syllable entropy")
    plt.ylabel("Conditional entropy")
    plt.title("Entropy-based clustering")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def accumulate_features(folder, agent_prefix):
    all_X = []
    all_labels = []

    for path in glob.glob(f"{folder}/*.txt"):
        X, _, _, labels = process_transcript(
            path,
            agent_prefix=agent_prefix
        )
        all_X.append(X)
        all_labels.append(labels)

    X_all = np.vstack(all_X)
    labels_all = np.concatenate(all_labels)

    return X_all, labels_all

def write_scalar_results(path, model, rounds, summaries):
    csr_vals = np.array([s["csr"] for s in summaries])
    ent_vals = np.array([s["entropy"] for s in summaries])
    cond_vals = np.array([s["cond_entropy"] for s in summaries])
    base_vals = np.array([s["baseline_entropy"] for s in summaries])

    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\nMODEL: {model} | ROUNDS: {rounds}\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"CSR: mean={csr_vals.mean():.3f}, std={csr_vals.std():.3f}\n"
        )
        f.write(
            f"Syllable entropy: mean={ent_vals.mean():.3f}, "
            f"std={ent_vals.std():.3f}\n"
        )
        f.write(
            f"Baseline entropy: mean={base_vals.mean():.3f}, "
            f"std={base_vals.std():.3f}\n"
        )
        f.write(
            f"Entropy - baseline: "
            f"{(ent_vals.mean() - base_vals.mean()):.3f}\n"
        )
        f.write(
            f"Conditional entropy: mean={cond_vals.mean():.3f}, "
            f"std={cond_vals.std():.3f}\n"
        )

def write_scalar_results_csv(path, model, rounds, summaries):
    csr_vals = np.array([s["csr"] for s in summaries])
    ent_vals = np.array([s["entropy"] for s in summaries])
    cond_vals = np.array([s["cond_entropy"] for s in summaries])
    base_vals = np.array([s["baseline_entropy"] for s in summaries])

    header = not os.path.exists(path)

    with open(path, "a", encoding="utf-8") as f:
        if header:
            f.write(
                "model,rounds,csr_mean,csr_std,"
                "entropy_mean,entropy_std,"
                "baseline_mean,baseline_std,"
                "entropy_minus_baseline,"
                "cond_entropy_mean,cond_entropy_std\n"
            )

        f.write(
            f"{model},{rounds},"
            f"{csr_vals.mean():.3f},{csr_vals.std():.3f},"
            f"{ent_vals.mean():.3f},{ent_vals.std():.3f},"
            f"{base_vals.mean():.3f},{base_vals.std():.3f},"
            f"{(ent_vals.mean() - base_vals.mean()):.3f},"
            f"{cond_vals.mean():.3f},{cond_vals.std():.3f}\n"
        )



# ----------------------------
# 8. Single-run processing
# ----------------------------

def process_transcript(path, agent_prefix):
    turns = load_agent_turns(path, agent_prefix)
    X, all_syll = extract_features(turns)

    if len(X) == 0:
        raise ValueError(f"No agent turns found in {path}")

    inventory = extract_canonical_inventory(all_syll)
    baseline = reduplicative_baseline_entropy(X[:, 3], inventory)

    labels = cluster_turns(X)

    summary = {
        "csr": X[:, 0].mean(),
        "entropy": X[:, 1].mean(),
        "cond_entropy": X[:, 2].mean(),
        "baseline_entropy": baseline.mean()
    }

    return X, X[:, 1], summary, labels


# ----------------------------
# 9. Multi-run aggregation
# ----------------------------

def run_batch(folder):
    entropy_traces = []
    summaries = []

    for path in glob.glob(f"{folder}/*.txt"):
        X, ent, summ, _ = process_transcript(path)
        entropy_traces.append(ent)
        summaries.append(summ)

    entropy_traces = np.stack(entropy_traces)

    mean_entropy = entropy_traces.mean(axis=0)
    std_entropy = entropy_traces.std(axis=0)

    return mean_entropy, std_entropy, summaries


# ----------------------------
# 10. Main
# ----------------------------

def run_all_conditions(root="logs"):
    MODEL_PREFIX = {
        "claude": "anthropic/claude-3.5-haiku",
        "gemini": "google/gemini-2.5-flash",
        "openai": "openai/gpt-4o-mini"
    }

    results = {}

    for model in ["claude", "gemini", "openai"]:
        prefix = MODEL_PREFIX[model]

        for rounds in ["50", "100", "200"]:
            folder = f"{root}/{model}/{rounds}"

            # ---- Batch processing ----
            entropy_traces = []
            summaries = []

            for path in glob.glob(f"{folder}/*.txt"):
                X, ent, summ, _ = process_transcript(
                    path,
                    agent_prefix=prefix
                )
                entropy_traces.append(ent)
                summaries.append(summ)

            entropy_traces = np.stack(entropy_traces)
            mean_e = entropy_traces.mean(axis=0)
            std_e = entropy_traces.std(axis=0)

            # ---- Save mean entropy plot ----
            baseline_mean = np.mean([s["baseline_entropy"] for s in summaries])

            # plot_mean_entropy(
            #     mean_e,
            #     std_e,
            #     baseline_mean,
            #     f"plots/mean_entropy/{model}_{rounds}.png"
            # )

            # ---- Representative cluster plot ----
            X_all, labels_all = accumulate_features(folder, prefix)

            # plot_entropy_clusters(
            #     X_all,
            #     labels_all,
            #     f"plots/clusters/{model}_{rounds}.png"
            # )

            # ---- Store results ----
            results[(model, int(rounds))] = {
                "mean_entropy": mean_e,
                "std_entropy": std_e,
                "summaries": summaries
            }

            write_scalar_results(
                path="results/scalar_statistics.txt",
                model=model,
                rounds=rounds,
                summaries=summaries
            )

            write_scalar_results_csv(
                path="results/scalar_statistics.csv",
                model=model,
                rounds=rounds,
                summaries=summaries
            )

    return results


if __name__ == "__main__":
    results = run_all_conditions("logs")
