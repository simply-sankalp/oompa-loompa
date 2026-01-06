import re
import pandas as pd

# Update this with the actual file path
input_file = "anthropic_claude-3.5-haiku_metrics.txt"
output_file = "ganthropic_claude-3.5-haiku_metrics.csv"

# Regex patterns for extracting metrics
patterns = {
    "TVR": r"Turn Validity Rate \(TVR\): ([\d\.\-]+)",
    "CSR": r"Conversation Success Rate \(CSR\): ([\d\.\-]+)",
    "AC": r"Adjacency Compliance \(AC\): ([\d\.\-]+)",
    "FR": r"Feedback Responsiveness \(FR\): ([\d\.\-]+)",
    "CE": r"Completion Efficiency \(CE\): ([\d\.\-]+)",
    "TTFK": r"Time to First Positive Feedback \(TTFK\): ([\d\.\-]+)",
}

data = []

with open(input_file, "r") as f:
    content = f.read()

runs = content.strip().split("----------------------------------------")

for run in runs:
    if not run.strip():
        continue

    run_data = {}
    run_match = re.search(r"Run (\d+)", run)
    if run_match:
        run_data["Run"] = int(run_match.group(1))

    for metric, pattern in patterns.items():
        match = re.search(pattern, run)
        if match:
            run_data[metric] = float(match.group(1))

    data.append(run_data)

df = pd.DataFrame(data).sort_values("Run")
df.to_csv(output_file, index=False)

print(f"CSV saved to {output_file}")
print(df.head())
