import pandas as pd
import matplotlib.pyplot as plt

filename = "feedback.csv"
df = pd.read_csv(filename)

agree_columns = {
    "Q1: The goals of the lab and homework were clearly defined and communicated": "Goals were clear(Q1)",
    "Q2: The instructions of the lab and homework were clearly defined and easy to follow": "Instructions were clear (Q2)",
    "Q5: The technical setup process was easy": "Setup was easy (Q5)",
    "Q7: The concept of metamorphic testing was understandable after completing this lab": "MT was understandable (Q7)",
    "Q9: Overall, what I learned in the lab is relevant for working in the software industry": "Lab was industry relevant (Q9)"
}

df.rename(columns=agree_columns, inplace=True)
clean_columns = list(agree_columns.values())

counts = df[clean_columns].apply(lambda col: col.value_counts(normalize=True) * 100).fillna(0)
response_order = ["Disagree", "Somewhat disagree", "Somewhat agree", "Agree"]
counts = counts.reindex(response_order)
percentages = counts.T
colors = {
    "Disagree": "#f94144", "Somewhat disagree": "#FD9F2C", "Somewhat agree": "#F1E868","Agree": "#90be6d"}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 16

# Agree/disagree chart
plt.figure(figsize=(12, 6))
bottom = None
for response in response_order:
    plt.bar(percentages.index, percentages[response], label=response,
            bottom=bottom, color=colors[response])
    bottom = percentages[response] if bottom is None else bottom + percentages[response]

plt.title("Response Distribution by Question", fontsize=17)
plt.ylabel("Percentage of Responses (%)", fontsize=16)
plt.xlabel("Question", fontsize=16)
plt.xticks(rotation=30, ha="right", fontsize=16)
plt.yticks(fontsize=14)
plt.ylim(0, 100)
plt.legend(title="Response", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14)
plt.tight_layout()
plt.show()

# Q6 plot
time_order = ["Under 2 hours", "2-5 hours", "5-10 hours", "Over 10 hours"]
q6_counts = df["Q6: How long did completing the homework take you?"].value_counts(normalize=True) * 100
q6_counts = q6_counts.reindex(time_order).dropna()
plt.figure(figsize=(8, 4))
plt.bar(q6_counts.index, q6_counts.values, color="#5B9BD5", width=0.6)
plt.title("Homework Completion Time", fontsize=16)
plt.ylabel("Percentage of Responses (%)", fontsize=16)
plt.xlabel("Time Spent", fontsize=14)
plt.xticks(rotation=0, ha="center", fontsize=16)
plt.yticks(fontsize=14)
plt.ylim(0, max(q6_counts.values) + 5)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Q4 plot
difficulty_map = {
    "Easier than previous assignments": "Easier",
    "About the same difficulty": "Same",
    "Slightly more difficult than previous assignments": "Slightly harder",
    "Much more difficult than previous assignments": "Much harder"
}
df["Q4_Short"] = df["Q4: How was the difficulty of this homework assignment compared to the previous ones?"].map(difficulty_map)
difficulty_order = ["Easier", "Same", "Slightly harder", "Much harder"]
q4_counts = df["Q4_Short"].value_counts(normalize=True) * 100
q4_counts = q4_counts.reindex(difficulty_order).dropna()

plt.figure(figsize=(8, 4))
plt.bar(q4_counts.index, q4_counts.values, color="#5B9BD5", width=0.6)
plt.title("Homework Difficulty Compared to Previous", fontsize=16)
plt.ylabel("Percentage of Responses (%)", fontsize=16)
plt.xlabel("Perceived Difficulty", fontsize=16)
plt.xticks(rotation=0, ha="center", fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, max(q4_counts.values) + 5)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
