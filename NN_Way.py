import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import csiread

# Known corrupted files from the dataset documentation
BAD_FILES = [
    "user2-6-4-4-2-r1.dat", "user3-1-3-1-8-r5.dat",
    "user2-3-5-3-4-r4.dat", "user6-3-1-1-5-r5.dat",
    "user8-1-1-1-1-r5.dat", "user8-3-3-3-5-r2.dat",
    "user9-1-1-1-1-r1.dat"
]

# Hyperparameters
MAX_TIME_STEPS = 400  # Number of packets to use (our "image height")
SUBCARRIERS = 30      # Intel 5300 subcarriers.
RX_ANTENNAS = 3       # Receiving antennas (our "color channels")
# NUM_LOCATIONS will be determined dynamically

def extract_csi_cnn_features(file_path):
    """
    Parses a .dat file and formats it into a fixed-size 3D tensor:
    (Time_Steps, Subcarriers, Rx_Antennas)
    """
    try:
        csidata = csiread.Intel(file_path)
        csidata.read()
        csi_complex = csidata.get_scaled_csi()

        if csi_complex is None or csi_complex.size == 0:
            return None

        amplitude = np.abs(csi_complex)

        # csiread.Intel typically returns (packets, subcarriers, rx_antennas, tx_antennas)
        # For this dataset, the shape is (packets, 30, 3, 2)
        # I need to end up with (Packets, Subcarriers, Rx_Antennas)

        if amplitude.ndim == 4 and amplitude.shape[3] >= 1:
            # Select the first Tx antenna (index 0) if multiple are present
            # This makes the shape: (packets, subcarriers, rx_antennas)
            amplitude = amplitude[:, :, :, 0]
        elif amplitude.ndim != 3:
            raise ValueError(f"Unexpected CSI amplitude dimensions: {amplitude.shape} for file {file_path}. Expected 3 or 4 dimensions.")

        # Now `amplitude` should be (Packets, Subcarriers, Rx_Antennas)
        # This is the "image" shape for the CNN: (Time_Steps, Width, Channels) where Width=Subcarriers, Channels=Rx_Antennas

        # Strict Min-Max Normalization per packet
        # Normalization should be done PER PACKET (across subcarriers and Rx antennas)
        packet_max = np.max(amplitude, axis=(1, 2), keepdims=True)
        packet_min = np.min(amplitude, axis=(1, 2), keepdims=True)

        # Avoid division by zero if all values in a packet are identical
        denominator = (packet_max - packet_min)
        amplitude = np.where(denominator == 0, 0, (amplitude - packet_min) / (denominator + 1e-8))

        # Pad or Truncate the time dimension to MAX_TIME_STEPS
        current_steps = amplitude.shape[0]

        if current_steps >= MAX_TIME_STEPS:
            # Truncate
            formatted_csi = amplitude[:MAX_TIME_STEPS, :, :]
        else:
            # Pad with zeros
            pad_width = ((0, MAX_TIME_STEPS - current_steps), (0, 0), (0, 0))
            formatted_csi = np.pad(amplitude, pad_width, mode='constant', constant_values=0)

        # Ensure the final shape matches the expected input for the CNN
        if formatted_csi.shape != (MAX_TIME_STEPS, SUBCARRIERS, RX_ANTENNAS):
             print(f"Warning: Final CSI features shape mismatch for {file_path}. Expected {(MAX_TIME_STEPS, SUBCARRIERS, RX_ANTENNAS)}, got {formatted_csi.shape}. Returning None.")
             return None

        return formatted_csi

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def build_cnn_model(input_shape, num_classes):
    """
    This function sets up our Convolutional Neural Network (CNN) architecture.
    It's designed to process the 3D 'images' we created from the CSI data.
    """
    model = models.Sequential([
        # Input_shape is (MAX_TIME_STEPS, SUBCARRIERS, RX_ANTENNAS) i.e. (400, 30, 3)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape), # Using (3,3) kernel as subcarriers is 30
        layers.MaxPooling2D((2, 2)), # Using (2,2) pooling as subcarriers is 30
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(), # This will average over (Time, Subcarriers) dimensions

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

  
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
   
    dataset_path = "/content/extracted_data/20181128/20181128/"
    file_pattern = os.path.join(dataset_path, "**/*.dat")

    X = []
    y = []

    print("Extracting 2D CSI matrices... This will take a few minutes.")
    filepaths = glob.glob(file_pattern, recursive=True)

    for file_path in filepaths:
        filename = os.path.basename(file_path)
        if filename in BAD_FILES:
            continue

        parts = filename.replace('.dat', '').split('-')
        if len(parts) == 6:
            try:
                # Widar locations are 1-8. Neural networks prefer 0-indexed labels (0-7).
                location_label = int(parts[2]) - 1

                features = extract_csi_cnn_features(file_path)
                if features is not None and features.shape == (MAX_TIME_STEPS, SUBCARRIERS, RX_ANTENNAS):
                    X.append(features)
                    y.append(location_label)
            except Exception as e:
                print(f"Skipping file {filename} due to error parsing label: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset Shape: {X.shape}") # Should be (Samples, 400, 30, 3)

    if len(X) == 0:
        print("No valid data found.")
        return

    num_unique_locations = len(np.unique(y))
    print(f"Detected {num_unique_locations} unique locations.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nBuilding CNN...")
    input_shape = (MAX_TIME_STEPS, SUBCARRIERS, RX_ANTENNAS)
    model = build_cnn_model(input_shape, num_unique_locations)

    print("Training CNN...")
    # Train for 20 epochs with a 20% validation split during training
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    print("\n--- Final Evaluation on Test Set ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Generate a detailed classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    # Adding 1 back to labels so they match the physical rooms (1-8)
    # Ensure target_names matches the actual number of classes
    print(classification_report(y_test, y_pred, target_names=[f"Loc {i+1}" for i in range(num_unique_locations)]))

if __name__ == "__main__":
    main()
