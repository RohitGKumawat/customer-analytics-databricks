# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "statsmodels",
# ]
# ///
# MAGIC %md
# MAGIC # A/B Test Analysis: Cookie Cats (gate_30 vs gate_40)
# MAGIC Design the test (hypothesis, metric, α, MDE, power) before looking at results, check for a Sample Ratio
# MAGIC Mismatch, then compare 7-day retention with a two-proportion z-test and a bootstrap check.

# COMMAND ----------

# MAGIC %pip install statsmodels

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the data

# COMMAND ----------

# Read the CSV file from the Unity Catalog volume
file_path = "/Volumes/workspace/default/cc_a_b_test/cookie_cats.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

print(f"Players: {df.count():,}")
df.printSchema()
display(df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test design (set before looking at results)

# COMMAND ----------

# ---------------------------------------------------------------------------
# A/B Test Design — Cookie Cats (gate_30 vs gate_40)
# ---------------------------------------------------------------------------

# Hypotheses (two-sided test)
#   H0 (null):       7-day retention is the same for gate_30 and gate_40
#                    (p_gate30  ==  p_gate40)
#   H1 (alternative): 7-day retention differs between gate_30 and gate_40
#                    (p_gate30  !=  p_gate40)

# Metric
#   Primary metric: 7-day retention rate
#   Defined as the proportion of players who returned to the game on
#   day 7 after install  (column: retention_7, values True/False).
#   This is a binary per-user metric, so the appropriate test is a
#   two-proportion z-test (or equivalently a chi-square test of independence).

# Significance level
alpha = 0.05        # 5 % — probability of a false positive (Type I error)

# Minimum Detectable Effect (MDE)
#   The smallest absolute change in 7-day retention we want to reliably detect.
mde = 0.01           # 1 percentage point (e.g. 18 % -> 19 %)

# Power (complementary to Type II error)
#   Standard convention; probability of correctly rejecting H0 when H1 is true.
power = 0.80

print("""
A/B Test Summary — Cookie Cats
-----------------------------------------------
Hypothesis      : H0: p_gate30 == p_gate40
                  H1: p_gate30 != p_gate40
Metric          : 7-day retention rate (retention_7)
Alpha (a)       : 0.05
MDE             : 0.01 (1 percentage point)
Power (1-B)     : 0.80
Test type       : Two-sided, two-proportion z-test
-----------------------------------------------
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Power analysis: required sample size per group

# COMMAND ----------

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from pyspark.sql import functions as F

# Baseline retention rate for gate_30 (control)
p_baseline = df.filter("version = 'gate_30'").select(
    (F.sum(F.col("retention_7").cast("int")) / F.count("*")).alias("rate")
).first()["rate"]

# Effect size for two-proportion test (Cohen's h)
es = proportion_effectsize(p_baseline, p_baseline - mde)

# Required sample size per group (two-sided test)
n_required = NormalIndPower().solve_power(
    effect_size=abs(es), alpha=alpha, power=power
)

# Actual sample sizes per group
actual_counts = (
    df.groupBy("version")
      .agg(F.count("*").alias("n"))
      .orderBy("version")
      .collect()
)
actual_dict = {row["version"]: row["n"] for row in actual_counts}

print("=" * 55)
print("Power Analysis — Sample Size per Group")
print("=" * 55)
print(f"Baseline retention (gate_30): {p_baseline:.4f}")
print(f"MDE:                        {mde:.4f}  (1 pp)")
print(f"Effect size (Cohen's h):    {abs(es):.4f}")
print(f"Alpha:                      {alpha}")
print(f"Power:                      {power}")
print(f"Required n per group:      {int(round(n_required)):,}")
print("-" * 55)
for v in sorted(actual_dict):
    print(f"Actual n  ({v}):          {actual_dict[v]:,}")
min_actual = min(actual_dict.values())
ratio = min_actual / n_required
print("-" * 55)
print(f"Smallest actual group:      {min_actual:,}")
print(f"Required per group:         {int(round(n_required)):,}")
if min_actual >= n_required:
    print(f"Ratio (actual / required):  {ratio:.1f}x  -> well powered")
else:
    print(f"Ratio (actual / required):  {ratio:.1f}x  -> underpowered")
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sample Ratio Mismatch (SRM) check

# COMMAND ----------

from scipy.stats import chisquare

# ---------------------------------------------------------------------------
# A 50/50 split should give roughly equal counts in each group. If the
# observed split deviates significantly from 50/50, a Sample Ratio Mismatch
# has occurred and the experiment's randomisation may be broken.
#
# One-sample chi-square goodness-of-fit test:
#   H0: observed counts follow the expected 50/50 proportions
#   H1: observed counts do not follow the expected 50/50 proportions
# ---------------------------------------------------------------------------

observed = [actual_dict[v] for v in sorted(actual_dict)]
labels   = sorted(actual_dict)
total    = sum(observed)
expected = [total / 2, total / 2]

chi2_stat, srm_p_value = chisquare(f_obs=observed, f_exp=expected)
srm_detected = srm_p_value < alpha

print("=" * 55)
print("Sample Ratio Mismatch (SRM) Check")
print("=" * 55)
for label, obs, exp in zip(labels, observed, expected):
    print(f"  {label}: observed = {obs:,}  expected = {exp:,.1f}  ({obs/total*100:.2f}%)")
print("-" * 55)
print(f"Chi-square statistic : {chi2_stat:.4f}")
print(f"p-value              : {srm_p_value:.6f}")
print(f"Alpha                : {alpha}")
print("-" * 55)
if srm_detected:
    print("Result: p < alpha  ->  SRM DETECTED")
    print("The split is NOT 50/50. The dataset has no pre-assignment attributes")
    print("to diagnose the cause, so the results below are treated as directional.")
else:
    print("Result: p >= alpha  ->  No SRM detected")
    print("The split is consistent with a 50/50 allocation.")
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Two-proportion z-test + bootstrap check

# COMMAND ----------

import numpy as np
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

# ---------------------------------------------------------------------------
#  H0:  p_gate30 == p_gate40
#  H1:  p_gate30 != p_gate40
#  Test: two-sided two-proportion z-test (pooled variance)
# ---------------------------------------------------------------------------

agg = (
    df.groupBy("version")
      .agg(
          F.sum(F.col("retention_7").cast("int")).alias("successes"),
          F.count("*").alias("n")
      )
      .collect()
)
row_g30 = [r for r in agg if r["version"] == "gate_30"][0]
row_g40 = [r for r in agg if r["version"] == "gate_40"][0]

counts = np.array([row_g30["successes"], row_g40["successes"]])
nobs   = np.array([row_g30["n"], row_g40["n"]])

p_g30 = counts[0] / nobs[0]
p_g40 = counts[1] / nobs[1]
diff  = p_g30 - p_g40          # gate_30 minus gate_40

# --- Two-proportion z-test (pooled) ---
z_stat, p_value = proportions_ztest(counts, nobs, alternative="two-sided")

# --- 95% confidence interval for the difference (Newcombe score method) ---
low_ci, high_ci = confint_proportions_2indep(
    count1=counts[0], nobs1=nobs[0],
    count2=counts[1], nobs2=nobs[1],
    compare="diff", alpha=alpha, method="score"
)

# --- Bootstrap check (percentile CI for the difference) ---
np.random.seed(42)
n_boot = 10_000

pandas_df = df.select("version", "retention_7").toPandas()
g30 = pandas_df.loc[pandas_df["version"] == "gate_30", "retention_7"].astype(int).values
g40 = pandas_df.loc[pandas_df["version"] == "gate_40", "retention_7"].astype(int).values

boot_diffs = np.empty(n_boot)
for i in range(n_boot):
    b30 = np.random.choice(g30, size=len(g30), replace=True)
    b40 = np.random.choice(g40, size=len(g40), replace=True)
    boot_diffs[i] = b30.mean() - b40.mean()

boot_ci = np.percentile(boot_diffs, [2.5, 97.5])

print("=" * 60)
print("Two-Proportion Z-Test — 7-day Retention (gate_30 vs gate_40)")
print("=" * 60)
print(f"gate_30 : n = {nobs[0]:,}  |  retained = {counts[0]:,}  |  rate = {p_g30:.4f}")
print(f"gate_40 : n = {nobs[1]:,}  |  retained = {counts[1]:,}  |  rate = {p_g40:.4f}")
print("-" * 60)
print(f"Observed difference (gate_30 - gate_40):  {diff:.4f}  ({diff*100:.2f} pp)")
print(f"z-statistic :  {z_stat:.4f}")
print(f"p-value     :  {p_value:.6f}")
print(f"Alpha       :  {alpha}")
print("-" * 60)
print("95% Confidence Interval for the difference (Newcombe score):")
print(f"  [{low_ci:.4f},  {high_ci:.4f}]   ({low_ci*100:.2f} pp,  {high_ci*100:.2f} pp)")
print("-" * 60)
print("Bootstrap check (10 000 resamples, percentile CI):")
print(f"  Mean boot difference :  {boot_diffs.mean():.4f}")
print(f"  95% bootstrap CI     :  [{boot_ci[0]:.4f},  {boot_ci[1]:.4f}]")
print("-" * 60)
if p_value < alpha:
    print("Decision: p < alpha  ->  REJECT H0")
    print("There IS a statistically significant difference in 7-day retention.")
else:
    print("Decision: p >= alpha  ->  FAIL TO REJECT H0")
    print("There is NO statistically significant difference in 7-day retention.")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Chart: 7-day retention with 95% confidence intervals

# COMMAND ----------

import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

# 95% Confidence intervals per group (Wilson score method)
ci_g30 = proportion_confint(counts[0], nobs[0], alpha=0.05, method="wilson")
ci_g40 = proportion_confint(counts[1], nobs[1], alpha=0.05, method="wilson")

groups = ["gate_30", "gate_40"]
retention_rates = [p_g30, p_g40]
errors = np.array([[p_g30 - ci_g30[0], p_g40 - ci_g40[0]],
                   [ci_g30[1] - p_g30, ci_g40[1] - p_g40]])
colors = ["#1f77b4", "#ff7f0e"]
x_pos = np.arange(len(groups))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x_pos, retention_rates, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
ax.errorbar(x_pos, retention_rates, yerr=errors, fmt="none",
            ecolor="black", capsize=10, capthick=2, linewidth=2)

# Value labels on bars
for bar, rate, ci in zip(bars, retention_rates, [ci_g30, ci_g40]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f"{rate*100:.2f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2., height - 0.01,
            f"95% CI:\n[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]",
            ha="center", va="top", fontsize=9, color="white", fontweight="bold")

ax.set_xlabel("Version (Gate Level)", fontsize=12, fontweight="bold")
ax.set_ylabel("7-Day Retention Rate", fontsize=12, fontweight="bold")
ax.set_title("A/B Test Results: 7-Day Retention by Gate Level\n(with 95% Confidence Intervals)",
             fontsize=14, fontweight="bold", pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(groups, fontsize=11)
ax.set_ylim(0, max(retention_rates) * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Significance bracket
if p_value < alpha:
    y_max = max(ci_g30[1], ci_g40[1]) + 0.01
    ax.plot([0, 1], [y_max, y_max], "k-", linewidth=1.5)
    ax.plot([0, 0], [y_max - 0.002, y_max], "k-", linewidth=1.5)
    ax.plot([1, 1], [y_max - 0.002, y_max], "k-", linewidth=1.5)
    ax.text(0.5, y_max + 0.003, f"p = {p_value:.4f}",
            ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Business recommendation

# COMMAND ----------

significant      = p_value < alpha
absolute_diff_pp = abs(diff) * 100
relative_change  = abs(diff) / p_g30 * 100

print("=" * 65)
print("Business Recommendation — Cookie Cats gate_30 vs gate_40")
print("=" * 65)

print("\n1. VALIDITY")
print("-" * 65)
print(f"   Required n per group  : {int(round(n_required)):,}  (actual: {min_actual:,}+, {ratio:.1f}x)")
print(f"   SRM check p-value     : {srm_p_value:.4f}  ->  {'SRM DETECTED' if srm_detected else 'no SRM'}")

print("\n2. STATISTICAL SIGNIFICANCE")
print("-" * 65)
print(f"   z-statistic           : {z_stat:.4f}")
print(f"   p-value               : {p_value:.6f}  ->  {'significant' if significant else 'not significant'} at alpha = {alpha}")

print("\n3. PRACTICAL SIGNIFICANCE")
print("-" * 65)
print(f"   gate_30 retention     : {p_g30*100:.2f}%")
print(f"   gate_40 retention     : {p_g40*100:.2f}%")
print(f"   Absolute difference   : {absolute_diff_pp:.2f} pp  ({relative_change:.1f}% relative)")
print(f"   95% CI (difference)   : [{low_ci*100:.2f} pp, {high_ci*100:.2f} pp]")
print(f"   95% bootstrap CI      : [{boot_ci[0]*100:.2f} pp, {boot_ci[1]*100:.2f} pp]")
print(f"   Designed MDE          : {mde*100:.2f} pp  ->  effect is {'at or above' if absolute_diff_pp >= mde*100 else 'below'} the MDE")

print("\n4. RECOMMENDATION")
print("-" * 65)
if significant and diff > 0:
    print("   >>> KEEP gate_30. Moving the gate to level 40 lowered 7-day retention")
    print(f"       by {absolute_diff_pp:.2f} pp (p = {p_value:.4f}), and retention is a leading")
    print("       indicator of player lifetime value.")
elif significant and diff < 0:
    print("   >>> ROLL OUT gate_40. Moving the gate to level 40 raised 7-day retention")
    print(f"       by {absolute_diff_pp:.2f} pp (p = {p_value:.4f}).")
else:
    print("   >>> DO NOT SHIP gate_40 on this evidence. The difference is not")
    print("       statistically significant and the CI includes zero.")
if srm_detected:
    print("\n   Caveat: an SRM was detected, so treat this result as directional and")
    print("   investigate the assignment pipeline before a final decision.")

print("\n   Next steps:")
print("     - Investigate the cause of the sample ratio imbalance.")
print("     - Review secondary metrics (retention_1, sum_gamerounds) before deciding.")
print("=" * 65)