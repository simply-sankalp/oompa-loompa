import re
import random
import numpy as np
from collections import Counter
from scipy.stats import entropy
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ----------------------------
# 1. Syllable segmentation
# ----------------------------

VOWELS = set("aeiou")

def cv_segment(token):
    """
    Segment token into CV/CVC pseudo-syllables.
    """
    token = token.lower()
    syllables = []
    i = 0
    n = len(token)

    while i < n:
        onset = ""
        nucleus = ""
        coda = ""

        while i < n and token[i] not in VOWELS and len(onset) < 2:
            onset += token[i]
            i += 1

        if i < n and token[i] in VOWELS:
            nucleus = token[i]
            i += 1
        else:
            break

        if i < n and token[i] not in VOWELS:
            coda = token[i]
            i += 1

        syllables.append(onset + nucleus + coda)

    return syllables


def extract_syllables(text):
    syllables = []
    for token in text.strip().split():
        syllables.extend(cv_segment(token))
    return syllables


def is_canonical_cv(syllable):
    """
    Canonical CV-like syllable:
    - exactly one vowel
    - starts with consonant
    """
    vowel_count = sum(c in VOWELS for c in syllable)
    return vowel_count == 1 and syllable[0] not in VOWELS


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
    """
    Approximate conditional entropy using syllable overlap.
    """
    if not curr or not prev:
        return 0.0

    overlap = set(curr) & set(prev)
    p = len(overlap) / len(set(curr))
    p = max(min(p, 0.999), 1e-6)

    return entropy([p, 1 - p], base=2)


# ----------------------------
# 3. Transcript parsing
# ----------------------------

def load_agent_turns(path, agent_prefix="google/gemini"):
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
    prev_syllables = None

    for turn in turns:
        syllables = extract_syllables(turn)
        all_syllables.extend(syllables)

        csr = canonical_syllable_ratio(syllables)
        h_syll = syllable_entropy(syllables)
        h_cond = conditional_entropy(syllables, prev_syllables)
        length = len(syllables)

        features.append([csr, h_syll, h_cond, length])
        prev_syllables = syllables

    return np.array(features), all_syllables


# ----------------------------
# 5. Reduplicative baseline
# ----------------------------

def extract_canonical_inventory(all_syllables):
    inventory = sorted({s for s in all_syllables if is_canonical_cv(s)})
    return inventory


def generate_reduplicative_utterance(length, inventory):
    syll = random.choice(inventory)
    return [syll] * length


def reduplicative_baseline_entropy(lengths, inventory, n_samples=1000):
    entropies = []

    for _ in range(n_samples):
        L = int(random.choice(lengths))
        syllables = generate_reduplicative_utterance(L, inventory)
        entropies.append(syllable_entropy(syllables))

    return np.array(entropies)


# ----------------------------
# 6. Clustering
# ----------------------------

def cluster_turns(X, k=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X_scaled)

    return labels


# ----------------------------
# 7. Plotting
# ----------------------------

def plot_entropy_over_time(entropies):
    plt.figure(figsize=(8, 4))
    plt.plot(entropies, marker="o")
    plt.xlabel("Turn index")
    plt.ylabel("Syllable entropy")
    plt.title("Syllable entropy across turns")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_entropy_clusters(X, labels):
    """
    X[:,1] = syllable entropy
    X[:,2] = conditional entropy
    """
    plt.figure(figsize=(6, 5))

    for c in sorted(set(labels)):
        idx = labels == c
        plt.scatter(
            X[idx, 1],
            X[idx, 2],
            label=f"Cluster {c}",
            alpha=0.7
        )

    plt.xlabel("Syllable entropy (H_syll)")
    plt.ylabel("Conditional entropy H(U_t | U_{t-1})")
    plt.title("Entropy-based clustering of conversational turns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------
# 8. Main pipeline
# ----------------------------

def main(path):
    turns = load_agent_turns(path)
    X, all_syllables = extract_features(turns)

    inventory = extract_canonical_inventory(all_syllables)

    real_entropies = X[:, 1]
    lengths = X[:, 3]

    baseline_entropies = reduplicative_baseline_entropy(
        lengths, inventory
    )

    labels = cluster_turns(X)

    print("\n=== Canonical syllable inventory ===")
    print(inventory)

    print("\n=== Reduplicative baseline comparison ===")
    print(f"Real mean entropy: {real_entropies.mean():.2f}")
    print(f"Baseline mean entropy: {baseline_entropies.mean():.2f}")

    print("\n=== Cluster summary ===")
    for c in sorted(set(labels)):
        idx = labels == c
        print(f"\nCluster {c}")
        print(f"Turns: {idx.sum()}")
        print(f"Mean CSR: {X[idx,0].mean():.2f}")
        print(f"Mean syllable entropy: {X[idx,1].mean():.2f}")
        print(f"Mean conditional entropy: {X[idx,2].mean():.2f}")

    plot_entropy_over_time(real_entropies)
    plot_entropy_clusters(X, labels)


if __name__ == "__main__":
    main("transcript.txt")
