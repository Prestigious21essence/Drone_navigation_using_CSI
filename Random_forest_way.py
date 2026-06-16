import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import csiread

# List of corrupted files to ignore based on the dataset bug notice
BAD_FILES = [
    "user2-6-4-4-2-r1.dat", "user3-1-3-1-8-r5.dat",
    "user2-3-5-3-4-r4.dat", "user6-3-1-1-5-r5.dat",
    "user8-1-1-1-1-r5.dat", "user8-3-3-3-5-r2.dat",
    "user9-1-1-1-1-r1.dat"
]

def extract_csi_features(file_path):
    """
    Reads the CSI .dat file and extracts basic statistical features.
    """
    # Widar 3.0 relies on Intel 5300 WiFi NICs
    csidata = csiread.Intel(file_path)
    csidata.read()

    # csi_complex shape is typically (packets, Tx, Rx, subcarriers)
    csi_complex = csidata.get_scaled_csi()

    if csi_complex.size == 0:
        return None

    # Extract the amplitude from the complex baseband signal
    amplitude = np.abs(csi_complex)

    # Calculate mean and variance across the temporal (packets) and spatial (Tx/Rx) dimensions
    # leaving us with a feature vector based strictly on the 30 subcarriers
    mean_amp = np.mean(amplitude, axis=(0, 1, 2))
    std_amp = np.std(amplitude, axis=(0, 1, 2))

    # Concatenate to create a 60-dimensional feature vector per file
    features = np.concatenate([mean_amp, std_amp])
    return features

def main():
    dataset_path = "/content/extracted_data/20181128/20181128/"
    file_pattern = os.path.join(dataset_path, "**", "*.dat") # Look for .dat files recursively

    X = []
    y = []

    print("Extracting features from CSI files...")
    for file_path in glob.glob(file_pattern, recursive=True):
        filename = os.path.basename(file_path)

        # Skip known empty/corrupted files
        if filename in BAD_FILES:
            continue

        # Widar3.0 Name Format: id-a-b-c-d-Rx.dat
        # Parameter 'b' is the torso location (our prediction target)
        parts = filename.replace('.dat', '').split('-')
        if len(parts) == 6:
            try:
                location_label = int(parts[2])

                features = extract_csi_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(location_label)
            except Exception as e:
                # Catch any errors during feature extraction or label parsing
                print(f"Skipping file {filename} due to error: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Successfully processed {len(X)} samples.")

    if len(X) == 0:
        print("No valid data found. Please verify the dataset_path and file names.")
        return

    # Split the dataset: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("\n--- Evaluation ---")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
