# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Customer Segmentation: RFM + KMeans
# MAGIC Online Retail II (UCI), sheet *Year 2009–2010*. Clean the transactions, build RFM features per customer,
# MAGIC choose k with silhouette + elbow (every run logged to MLflow), then profile and name the segments.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load, clean and save as a Delta table

# COMMAND ----------

from pyspark.sql.functions import col, trim

# Read the raw invoice data from the Unity Catalog volume
raw_df = spark.read.format("excel").option("headerRows", 1).load("/Volumes/workspace/default/online_retail_ii/online_retail_II.xlsx")

original_count = raw_df.count()

clean_df = (
    raw_df
    # Drop rows with no CustomerID (null or blank after trimming)
    .filter((col("Customer ID").isNotNull()) & (trim(col("Customer ID").cast("string")) != ""))
    # Drop cancelled invoices — InvoiceNo starts with "C"
    .filter(~col("Invoice").rlike("^C"))
    # Drop rows with negative quantities
    .filter(col("Quantity") >= 0)
    # Delta tables don't allow spaces in column names
    .withColumnRenamed("Customer ID", "CustomerID")
)

# Save once as a Delta table so later cells don't re-read the Excel file
clean_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.default.online_retail_clean")
clean_df = spark.table("workspace.default.online_retail_clean")

clean_count = clean_df.count()

print(f"Original rows : {original_count:,}")
print(f"Cleaned rows  : {clean_count:,}")
print(f"Rows dropped  : {original_count - clean_count:,}")

display(clean_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. RFM features per customer

# COMMAND ----------

from pyspark.sql.functions import col, lit, sum as _sum, countDistinct, max as _max, datediff, round as _round

# Reference date: the latest invoice date in the dataset
ref_date = clean_df.select(_max(col("InvoiceDate")).alias("max_date")).collect()[0]["max_date"]

rfm_df = (
    clean_df
    # Monetary = Quantity × Price per line item
    .withColumn("LineTotal", col("Quantity") * col("Price"))
    .groupBy("CustomerID")
    .agg(
        # Recency: days since the customer's most recent purchase
        datediff(
            lit(ref_date).cast("date"),
            _max(col("InvoiceDate")).cast("date")
        ).alias("Recency"),
        # Frequency: number of distinct invoices (orders)
        countDistinct("Invoice").alias("Frequency"),
        # Monetary: total spend across all line items
        _round(_sum("LineTotal"), 2).alias("Monetary"),
    )
)

print(f"Reference date (latest invoice): {ref_date}")
print(f"Total customers: {rfm_df.count():,}")
display(rfm_df.orderBy(col("Monetary").desc()).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Choose k: KMeans for k = 2..10, every run logged to MLflow

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import log1p, col
import matplotlib.pyplot as plt
import mlflow

# ── 1. Log-transform the RFM features (add 1 to avoid log(0)) ──────────────
rfm_log = rfm_df.withColumn("Recency_log", log1p(col("Recency"))) \
    .withColumn("Frequency_log", log1p(col("Frequency"))) \
    .withColumn("Monetary_log", log1p(col("Monetary")))

# ── 2. Assemble into a feature vector ──────────────────────────────────────
assembler = VectorAssembler(
    inputCols=["Recency_log", "Frequency_log", "Monetary_log"],
    outputCol="features_raw"
)
rfm_assembled = assembler.transform(rfm_log)

# ── 3. Scale the features (zero mean, unit variance) ───────────────────────
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)
scaler_model = scaler.fit(rfm_assembled)
rfm_scaled = scaler_model.transform(rfm_assembled)

# ── 4. Set up MLflow experiment (path built from the current user, no hard-coded email) ──
current_user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{current_user}/customer-analytics-databricks/rfm_kmeans_clustering")

# ── 5. Run KMeans for k = 2..10 and collect silhouette + WSSSE ──────────────
k_values = range(2, 11)
silhouette_scores = []
wssse_scores = []

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

for k in k_values:
    with mlflow.start_run(run_name=f"kmeans_k{k}"):
        kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="prediction")
        model = kmeans.fit(rfm_scaled)
        predictions = model.transform(rfm_scaled)

        silhouette = evaluator.evaluate(predictions)
        wssse = model.summary.trainingCost

        mlflow.log_params({
            "k": k,
            "seed": 42,
            "scaler": "StandardScaler",
            "scaler_withMean": True,
            "scaler_withStd": True,
            "distance_measure": "squaredEuclidean",
        })
        mlflow.log_metrics({
            "silhouette": silhouette,
            "inertia": wssse,
        })

        silhouette_scores.append(silhouette)
        wssse_scores.append(wssse)
        print(f"k={k:>2d}  |  Silhouette={silhouette:.4f}  |  WSSSE={wssse:,.2f}")

# ── 6. Plot silhouette scores and elbow (WSSSE) curve side by side ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(k_values), silhouette_scores, marker="o", linestyle="-")
axes[0].set_title("Silhouette Score by k")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Silhouette Score")
axes[0].set_xticks(list(k_values))
axes[0].grid(True)

axes[1].plot(list(k_values), wssse_scores, marker="o", linestyle="-")
axes[1].set_title("Elbow Plot (WSSSE by k)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("WSSSE")
axes[1].set_xticks(list(k_values))
axes[1].grid(True)

plt.tight_layout()

# ── 7. Log the evaluation plots as an MLflow artifact ──────────────────────
with mlflow.start_run(run_name="evaluation_plots"):
    mlflow.log_params({
        "scaler": "StandardScaler",
        "scaler_withMean": True,
        "scaler_withStd": True,
        "k_range": f"{min(k_values)}-{max(k_values)}",
    })
    mlflow.log_figure(fig, "kmeans_evaluation_plots.png")

plt.show()

# ── 8. Pick the best k by silhouette score ─────────────────────────────────
best_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]
best_silhouette = max(silhouette_scores)
print(f"\nBest k by silhouette score: k={best_k} (silhouette={best_silhouette:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Final model: profile and name the segments

# COMMAND ----------

from pyspark.sql.functions import col, avg, count, sum as _sum, round as _round
from pyspark.ml.clustering import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Train the final KMeans model with the best k ─────────────────────────
final_model = KMeans(k=best_k, seed=42, featuresCol="features", predictionCol="cluster").fit(rfm_scaled)
rfm_labeled = final_model.transform(rfm_scaled).select("CustomerID", "Recency", "Frequency", "Monetary", "cluster")

total_customers = rfm_labeled.count()
total_revenue = rfm_labeled.agg(_sum("Monetary")).first()[0]

# ── 2. Profile each cluster: average R, F, M, size and share of revenue ─────
profile_df = (
    rfm_labeled.groupBy("cluster")
    .agg(
        count("CustomerID").alias("num_customers"),
        _round(avg("Recency"), 1).alias("avg_recency"),
        _round(avg("Frequency"), 1).alias("avg_frequency"),
        _round(avg("Monetary"), 2).alias("avg_monetary"),
        _round(_sum("Monetary"), 2).alias("total_revenue"),
    )
    .withColumn("pct_customers", _round(col("num_customers") / total_customers * 100, 1))
    .withColumn("pct_revenue", _round(col("total_revenue") / total_revenue * 100, 1))
    .orderBy("cluster")
)

print("Cluster profiles:")
profile_df.show(truncate=False)

# ── 3. Name the segments from their profiles ────────────────────────────────
#    Cluster ids can change between runs, so names are assigned by spend
#    (highest first). For k = 2 the high-spend cluster is also the most recent
#    and most frequent — check the table above. Update the list if k changes.
segment_names = ["Active High-Value", "Lapsed Low-Value"]
assert best_k == len(segment_names), "Update segment_names to match the chosen k"

profiles_pd = profile_df.toPandas().sort_values("avg_monetary", ascending=False)
seg_map = spark.createDataFrame(
    pd.DataFrame({"cluster": profiles_pd["cluster"].astype(int).tolist(), "segment": segment_names})
)

profile_named = profile_df.join(seg_map, on="cluster", how="left").orderBy(col("avg_monetary").desc())
print("\nSegments:")
display(profile_named)

# ── 4. Attach segment name to every customer and save ───────────────────────
rfm_final = rfm_labeled.join(seg_map, on="cluster", how="left")
rfm_final.write.mode("overwrite").saveAsTable("workspace.default.customer_segments")
display(rfm_final.orderBy(col("Monetary").desc()).limit(20))

# ── 5. Chart: share of customers vs share of revenue per segment ────────────
summary_pd = profile_named.toPandas()
x = np.arange(len(summary_pd))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars_c = ax.bar(x - width / 2, summary_pd["pct_customers"], width, label="% of customers")
bars_r = ax.bar(x + width / 2, summary_pd["pct_revenue"], width, label="% of revenue")
ax.bar_label(bars_c, fmt="%.1f%%")
ax.bar_label(bars_r, fmt="%.1f%%")
ax.set_xticks(x)
ax.set_xticklabels(summary_pd["segment"])
ax.set_ylabel("Share (%)")
ax.set_ylim(0, 100)
ax.set_title("Customer Segments: Share of Customers vs Share of Revenue")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()