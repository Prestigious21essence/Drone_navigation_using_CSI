import os
import glob
import numpy as np
import scipy.linalg as la
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import csiread

BAD_FILES = [
    "user2-6-4-4-2-r1.dat", "user3-1-3-1-8-r5.dat",
    "user2-3-5-3-4-r4.dat", "user6-3-1-1-5-r5.dat",
    "user8-1-1-1-1-r5.dat", "user8-3-3-3-5-r2.dat",
    "user9-1-1-1-1-r1.dat"
]

def sanitize_phase(complex_csi):
    """
    Applies SpotFi's phase sanitization algorithm to remove timing offsets.
    Expects input shape: (Subcarriers, Antennas). Assumes at least 2 subcarriers.
    """
    raw_phase = np.angle(complex_csi)
    sanitized_phase = np.zeros_like(raw_phase)

    # Dynamically determine subcarrier indices based on actual data
    num_actual_subcarriers = complex_csi.shape[0]
    # Using arange for simplicity, assuming relative spacing is sufficient for polyfit
    subcarrier_indices = np.arange(num_actual_subcarriers)

    for ant in range(raw_phase.shape[1]):
        # Unwrap phase to remove pi/-pi jumps
        unwrapped = np.unwrap(raw_phase[:, ant])

        # SpotFi linear fit: find the slope (m) and intercept (c)
        m, c = np.polyfit(subcarrier_indices, unwrapped, 1)

        # Subtract the linear error
        sanitized_phase[:, ant] = unwrapped - (m * subcarrier_indices + c)

    # Reconstruct the complex CSI with the cleaned phase
    amplitude = np.abs(complex_csi)
    clean_csi = amplitude * np.exp(1j * sanitized_phase)
    return clean_csi

def music_1d(clean_csi, num_sources=2): # Changed default num_sources to 2 for 3 antennas
    """
    A 1D MUSIC algorithm to find the Angle of Arrival (AoA) across the antennas.
    Returns the MUSIC pseudospectrum which acts as our spatial feature.
    """
    # Number of antennas (Intel 5300 has 3) ***This is Rx antennas***
    M = clean_csi.shape[1]

    # Calculate the covariance matrix of the antennas
    # Average across all subcarriers to get a stable matrix
    R = np.zeros((M, M), dtype=complex)
    for sub in range(clean_csi.shape[0]):
        x = clean_csi[sub, :].reshape(-1, 1)
        R += x @ x.conj().T
    R = R / clean_csi.shape[0]

    # Eigenvalue Decomposition
    eigenvalues, eigenvectors = la.eigh(R)

    # Sort eigenvalues and separate the noise subspace
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    noise_subspace = eigenvectors[:, num_sources:]

    # If noise subspace is empty (e.g., M <= num_sources), fallback to arbitrary vector
    if noise_subspace.shape[1] == 0:
         # Fallback to taking the smallest eigenvector as noise if num_sources >= M
         noise_subspace = eigenvectors[:, -1:]

    # Search across angles from -90 to 90 degrees
    theta = np.linspace(-np.pi/2, np.pi/2, 90)
    P_music = np.zeros(len(theta))

    # Distance between antennas (usually half-wavelength, d = 0.5) ***Assumes uniform linear array***
    d = 0.5

    for i, angle in enumerate(theta):
        # Steering vector for the antenna array
        steering_vector = np.exp(-1j * 2 * np.pi * d * np.arange(M) * np.sin(angle)).reshape(-1, 1)

        # MUSIC spectrum calculation: 1 / (a^H * En * En^H * a)
        denom = steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector
        P_music[i] = 10 * np.log10(1.0 / np.abs(denom[0, 0]) + 1e-10)

    return P_music

def extract_spotfi_features(file_path):
    """
    Parses the file, sanitizes phase, runs MUSIC, and averages across time.
    """
    try:
        csidata = csiread.Intel(file_path)
        csidata.read()
        csi_complex = csidata.get_scaled_csi()

        if csi_complex is None or csi_complex.size == 0:
            print(f"Warning: {file_path} contains no valid CSI data after read.")
            return None

        # Original shape from csiread.Intel is typically (packets, subcarriers, rx_antennas, tx_antennas)
        # I need to process CSI for one Tx antenna, across all Rx antennas and subcarriers.
        if csi_complex.ndim == 4:
            # Select the first Tx antenna (index 0) to get (packets, subcarriers, rx_antennas)
            csi_processed = csi_complex[:, :, :, 0]
        elif csi_complex.ndim == 3:
            # If it's already 3D, assume it's (packets, subcarriers, rx_antennas)
            csi_processed = csi_complex
        else:
            print(f"Warning: Unexpected CSI shape {csi_complex.shape} for {file_path}. Skipping.")
            return None

        # Now csi_processed should be (Packets, Subcarriers, Rx_Antennas)

        # For speed in this demo, I will sample every 10th packet
        sampled_csi = csi_processed[::10, :, :]

        all_spectra = []
        if sampled_csi.shape[0] == 0:
            print(f"Warning: No packets sampled for {file_path} after downsampling.")
            return None

        for packet_idx in range(sampled_csi.shape[0]):
            packet_data = sampled_csi[packet_idx, :, :] # Shape: (Subcarriers, Antennas)

            # CRITICAL CHECK: Ensure enough subcarriers for phase sanitization
            if packet_data.shape[0] < 2: # np.polyfit(..., deg=1) needs at least 2 points
                print(f"Warning: Skipping packet {packet_idx} from {file_path} due to insufficient subcarriers ({packet_data.shape[0]}). Minimum 2 required for phase sanitization.")
                continue

            # 1. Sanitize Phase
            clean_packet = sanitize_phase(packet_data)

            # 2. Run MUSIC to get Angle of Arrival spectrum
            spectrum = music_1d(clean_packet, num_sources=2)
            all_spectra.append(spectrum)

        if not all_spectra: # If all packets were skipped or no spectra were generated
            print(f"Warning: No valid AoA spectra extracted for {file_path}.")
            return None

        # Average the AoA spectrum across all sampled packets to get the final room fingerprint
        final_feature = np.mean(np.array(all_spectra), axis=0)
        return final_feature

    except Exception as e:
        print(f"Error extracting SpotFi features from {file_path}: {e}")
        return None

def main():
    dataset_path = "/content/extracted_data/20181128/20181128/" 
    file_pattern = os.path.join(dataset_path, "**/*.dat")

    X, y = [], []

    print("Running SpotFi Phase Sanitization and MUSIC... This requires heavy computation.")
    filepaths = glob.glob(file_pattern, recursive=True)

    for file_path in filepaths:
        filename = os.path.basename(file_path)
        if filename in BAD_FILES:
            continue

        parts = filename.replace('.dat', '').split('-')
        if len(parts) == 6:
            try:
                # Widar locations are 1-8. Note: SpotFi often uses 1-indexed labels directly.
                location_label = int(parts[2])

                features = extract_spotfi_features(file_path)
                if features is not None: # Only append if feature extraction was successful
                    X.append(features)
                    y.append(location_label)
            except Exception as e:
                print(f"Skipping file {filename} due to error parsing label or features: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\nExtracted SpotFi AoA Spectra for {len(X)} samples.")

    if len(X) == 0:
        print("No valid data found.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Classifier on Geometric AoA Features...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("\n--- SpotFi Evaluation ---")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
