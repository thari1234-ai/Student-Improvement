import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ================= LOAD DATA =================
data = pd.read_csv("data/student_data.csv")

# ================= USER INPUT =================
name = input("Enter your Name: ")
roll_no = input("Enter your Roll Number: ")

student_df = data[data['student_id'] == int(roll_no)]
if student_df.empty:
    print(f"No data found for Roll No {roll_no}")
    exit()

# ================= FUNCTION =================
def calculate_improvement(student_df, degree=2):
    X = student_df[['week']].values
    y = student_df['test_score'].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    coef = model.coef_
    last_week = X.max()
    improvement_rate = coef[1] + 2 * coef[2] * last_week

    return improvement_rate, model, poly, X, y

def generate_reason(student_df, category):
    reasons = []
    # Compute averages
    hw_avg = student_df['homework_pct'].mean()
    att_avg = student_df['attendance_pct'].mean()
    extra_avg = student_df['extra_class_hours'].mean()
    
    # High Improvement
    if category == "High Improvement":
        if hw_avg >= 75:  # lowered threshold
            reasons.append("Consistently completes homework")
        if att_avg >= 85:  # lowered threshold
            reasons.append("Excellent attendance")
        if extra_avg >= 1.5:  # lowered threshold
            reasons.append("Participates in extra classes")
    # Moderate Improvement
    elif category == "Moderate Improvement":
        reasons.append("Moderate study habits or attendance")
    # Low Improvement
    else:
        if hw_avg < 65:
            reasons.append("Low homework completion")
        if att_avg < 80:
            reasons.append("Poor attendance")
        if extra_avg < 1:
            reasons.append("Rarely attends extra classes")
    
    # If nothing matched
    if not reasons:
        reasons.append("No specific reason detected")
    
    return ", ".join(reasons)

# ================= MODELING =================
rate, model, poly, X, y = calculate_improvement(student_df)

# Categorize improvement
if rate > 2:
    category = "High Improvement"
elif rate >= 1:
    category = "Moderate Improvement"
else:
    category = "Low Improvement"

# Generate reason
reason = generate_reason(student_df, category)

# ================= SAVE OUTPUT =================
output = pd.DataFrame({
    'student_id': [roll_no],
    'improvement_rate': [rate],
    'category': [category],
    'reason': [reason],
    'analyst_name': [name]
})
output.to_csv("improvement_rates.csv", index=False)

# ================= DISPLAY =================
print("\n======================================")
print(f"Name: {name}")
print(f"Roll No: {roll_no}")
print(f"Improvement Rate: {rate:.2f}")
print(f"Improvement Category: {category}")
print(f"Reason: {reason}")
print("======================================\n")

# ================= VISUALIZATION =================
# ================= VISUALIZATION =================
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(10,6), facecolor='#f0f0f0')  # Figure background color (light gray)

# Scatter + polynomial curve
plt.scatter(X, y, color='blue', label='Actual Scores', s=50)
plt.plot(X_plot, y_plot, color='red', label='Polynomial Fit', linewidth=3)

# Set axes background color
plt.gca().set_facecolor('#e6f2ff')  # light blue inside plot area

# Labels
plt.xlabel("Week", fontsize=14, fontweight='bold')
plt.ylabel("Test Score", fontsize=14, fontweight='bold')
plt.title(f"{name}'s Academic Improvement Curve\nRoll No: {roll_no} | Category: {category}", 
          fontsize=16, fontweight='bold')

# Plot reasons with bold
reason_lines = reason.split(", ")
for i, line in enumerate(reason_lines):
    plt.text(X.max() + 0.2, y.max() - i*3, line, fontsize=12, fontweight='bold', color='green')

# Legend
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

