import joblib
import numpy as np
from pathlib import Path
import warnings

# Suppress warnings from sklearn (optional but recommended)
warnings.filterwarnings('ignore')

def predict_segment(recency, frequency, monetary, discount):
    """
    Loads the models and transforms raw input to predict the customer segment.
    """
    try:
        # --- 1. Load Models ---
        BASE_DIR = Path.cwd()
        MODEL_DIR = BASE_DIR / "model"

        # Load the 4 transformer models
        recency_transformer = joblib.load(MODEL_DIR / "transformer_recency.joblib")
        frequency_transformer = joblib.load(MODEL_DIR / "transformer_frequency.joblib")
        monetary_transformer = joblib.load(MODEL_DIR / "transformer_monetary.joblib")
        discount_transformer = joblib.load(MODEL_DIR / "transformer_discount.joblib")
        
        # Load the K-Means model
        kmeans_model = joblib.load(MODEL_DIR / "kmeans_clustering_model.joblib")

    except FileNotFoundError as e:
        return f"Error: Model file not found. Ensure models are in {MODEL_DIR}\nDetail: {e}"
    except Exception as e:
        return f"Error loading models: {e}"

    try:
        # --- 2. Transform Input Data ---
        # NOTE: The model was trained on a DOLLAR scale (e.g., 7000), NOT Billions.
        # Extreme outlier inputs will be classified based on the learned scaler.
        
        # Reshape data to 2D array for PowerTransformer
        recency_arr = np.array([[recency]])
        frequency_arr = np.array([[frequency]])
        monetary_arr = np.array([[monetary]])
        discount_arr = np.array([[discount]])

        # Apply the saved transformations
        recency_transformed = recency_transformer.transform(recency_arr)
        frequency_transformed = frequency_transformer.transform(frequency_arr)
        monetary_transformed = monetary_transformer.transform(monetary_arr)
        discount_transformed = discount_transformer.transform(discount_arr)

        # --- 3. Combine Features for K-Means ---
        # Order must be identical to the training notebook
        feature_vector = np.concatenate([
            recency_transformed,
            frequency_transformed,
            monetary_transformed,
            discount_transformed
        ], axis=1)

        # --- 4. Predict Cluster ---
        cluster_id = kmeans_model.predict(feature_vector)[0]
        
        return cluster_id

    except Exception as e:
        return f"Error during prediction: {e}"

def map_cluster_to_name(cluster_id):
    """
    Maps the K-Means Cluster ID to a descriptive segment name.
    (MAPPING UPDATED based on your model's actual data)
    """
    # Based on notebook data & tests:
    # Cluster 0 = HIGHEST Monetary/Discount (Top Customer)
    # Cluster 3 = LOWEST Monetary/Frequency (Low Value)
    segment_map = {
        0: "Top Customer (Loyal, Big Spenders)",
        1: "High Value Customer (Active, Recent Buyer)",
        2: "Medium Value Customer",
        3: "Low Value / At-Risk Customer (Infrequent Buyer)"
    }
    return segment_map.get(cluster_id, f"Unknown Cluster ({cluster_id})")

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Superstore Customer Segment Predictor ---")
    print("IMPORTANT: Please enter Monetary & Discount values in Dollars (e.g., 5000.50)")

    try:
        # --- Get Input from User ---
        r = float(input("Enter Recency (days, e.g., 20): "))
        f = float(input("Enter Frequency (total orders, e.g., 10): "))
        m = float(input("Enter Monetary (total sales in $, e.g., 7000): "))
        d = float(input("Enter Discount Amount (total discount in $, e.g., 1100): "))

        # --- Make Prediction ---
        prediction = predict_segment(r, f, m, d)

        # Check if 'prediction' is an integer (success) or string (error)
        if isinstance(prediction, (int, np.integer)):
            segment_name = map_cluster_to_name(prediction)
            
            print("\n--- Prediction Result ---")
            print(f"Predicted Cluster ID: {prediction}")
            print(f"Customer Segment:     {segment_name}")
        else:
            # Print the error message
            print(f"\n{prediction}")

    except ValueError:
        print("\nError: Invalid input. Please enter numeric values only.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")