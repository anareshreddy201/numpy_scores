import numpy as np

# =========================================
# TASK 1 — Generate and Inspect the Data
# =========================================

np.random.seed(42)

# Generate random scores (5 students, 4 subjects)
scores = np.random.randint(50, 101, size=(5, 4))

print("Scores:\n", scores)

# 3rd student (index 2), 2nd subject (index 1)
print("\nScore of 3rd student in 2nd subject:", scores[2, 1])

# Last 2 students (all subjects)
print("\nScores of last 2 students:\n", scores[-2:, :])

# First 3 students, subjects 2 and 3 only
print("\nFirst 3 students, subjects 2 & 3:\n", scores[:3, 1:3])


# =========================================
# TASK 2 — Analyze with Broadcasting
# =========================================

# Column-wise mean (average per subject)
column_means = np.round(np.mean(scores, axis=0), 2)
print("\nColumn-wise Mean:", column_means)

# Add curve using broadcasting
curve = np.array([5, 3, 7, 2])
curved_scores = scores + curve

# Ensure no score exceeds 100
curved_scores = np.clip(curved_scores, None, 100)

print("\nCurved Scores:\n", curved_scores)

# Row-wise max (best subject per student)
row_max = np.max(curved_scores, axis=1)
print("\nBest score per student:", row_max)


# =========================================
# TASK 3 — Normalize and Identify
# =========================================

# Min-max normalization per row
row_min = np.min(curved_scores, axis=1, keepdims=True)
row_max = np.max(curved_scores, axis=1, keepdims=True)

normalized = (curved_scores - row_min) / (row_max - row_min)

print("\nNormalized Scores:\n", normalized)

# Find index of single highest normalized value
max_index = np.unravel_index(np.argmax(normalized), normalized.shape)

print("\nHighest normalized value at:")
print("Student index:", max_index[0])
print("Subject index:", max_index[1])

# Extract scores strictly above 90
above_90 = curved_scores[curved_scores > 90]

print("\nScores above 90:", above_90)