# Superstore Customer Segmentation Analysis

This project analyzes the Superstore dataset to perform customer segmentation. The segmentation is based on an RFM (Recency, Frequency, Monetary) model, modified with **Discount Amount** data, to group customers into distinct clusters.

The goal of this project is to identify different customer archetypes (e.g., "Top Customer," "Low Value") so that more targeted business strategies can be developed.

This analysis uses **K-Means Clustering** to create the segmentation model. The project includes:
1.  `notebook.ipynb`: The Jupyter Notebook containing all steps for analysis, preprocessing, model training, and visualization.
2.  `model/`: A directory containing the 4 `PowerTransformer` models and the 1 `KMeans` model saved as `.joblib` files.
3.  `pred.py`: A Python script to run live predictions on new data.

---

---

## How to Run Predictions

You can run the `pred.py` script to get a segment prediction for a new customer.

1.  **Clone the Repository**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd Superstore
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Prediction Script**
    ```bash
    python pred.py
    ```

4.  **Enter Customer Data**
    Follow the prompts in the terminal.
    **IMPORTANT:** This model was trained using **Dollar** values (e.g., 7000.50), not other currencies or large-scale numbers (e.g., 7,000,000). Please enter Monetary and Discount values on the correct scale.

    ```bash
    --- Superstore Customer Segment Predictor ---
    IMPORTANT: Please enter Monetary & Discount values in Dollars (e.g., 5000.50)
    Enter Recency (days, e.g., 20): 20
    Enter Frequency (total orders, e.g., 10): 10
    Enter Monetary (total sales in $, e.g., 7000): 7000
    Enter Discount Amount (total discount in $, e.g., 1100): 1100
    ```

---

## Analysis & Segment Results (from `random_state=99` Model)

Based on the data analysis from `notebook.ipynb`, the trained K-Means model (`random_state=99`) produces 4 customer clusters with the following characteristics:

**Insight**:
* **Cluster 0** is the customer group with the highest **Monetary** (average $5k-$7k) and **Discount** (average $500-$1k). This is the cluster with the **fewest** customers.
* **Cluster 1** is the customer group with the **best Recency** (most recent transaction, e.g., 19 days) and the **highest Frequency** (e.g., 9 transactions).
* **Cluster 3** is the customer group with the **worst Recency** (longest time since last transaction, e.g., 555 days) and the **lowest Frequency** (e.g., 3 transactions). This is the cluster with the **most** customers.
* **Cluster 2** is the *Medium/Low Value* customer group with Frequency and Discount that tend to be low.

**Result (Segment Names):**
* **Cluster 0 → Top Customers** (Loyal, Big Spenders, receive high discounts).
* **Cluster 1 → High Value Customers** (Very active, most recent transactions, high potential to become Top Customers).
* **Cluster 2 → Medium Value Customers** (Standard value).
* **Cluster 3 → Low Value / At-Risk Customers** (Rarely transact, haven't returned in a long time, low transaction value).
