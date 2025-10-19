
import time
import numpy as np
import acconeer.exptool as et
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ---- Breathing Rate Estimation Parameters ----
BREATHING_RATE_RANGE_BPM = [8, 20] # Expected breathing rate in breaths per minute

# ---- Tune these to your geometry ----
RANGE_INTERVAL = [0.2, 1.0]   # The range of distances to measure (in meters)
UPDATE_RATE_HZ = 10          # The update rate of the sensor

# ---- Presence Detection Thresholds ----
PRESENCE_AMP_THRESHOLD = 0.01 # Amplitude threshold for presence detection
PRESENCE_DEBOUNCE_S = 2       # Seconds to wait before declaring presence/absence

# ---- Time Series and Filtering ----
TIME_SERIES_DURATION_S = 15 # Duration of the time series for FFT (in seconds)

def get_breathing_rate_range_hz():
    return [r / 60 for r in BREATHING_RATE_RANGE_BPM]

def bandpass_filter(data, fs):
    lowcut, highcut = get_breathing_rate_range_hz()
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, data)

def main():
    args = et.a111.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    client = et.a111.Client(**et.a111.get_client_args(args))

    cfg = et.a111.IQServiceConfig()
    cfg.sensor = args.sensors
    cfg.range_interval = RANGE_INTERVAL
    cfg.update_rate = UPDATE_RATE_HZ
    
    client.connect() 
    
    si = client.setup_session(cfg)
    print("Session info:", si)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Filtered Breathing Waveform plot
    line1, = ax1.plot([], [])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Phase")
    ax1.set_title("Filtered Breathing Waveform")

    # Power Spectrum plot
    line2, = ax2.plot([], [])
    ax2.set_xlabel("Breathing Rate (BPM)")
    ax2.set_ylabel("Power")
    ax2.set_title("Breathing Rate Power Spectrum")
    fig.tight_layout(pad=3.0)

    time_series_len = int(TIME_SERIES_DURATION_S * UPDATE_RATE_HZ)
    phase_history = deque(maxlen=time_series_len)
    time_history = deque(maxlen=time_series_len)

    present = False
    presence_debounce_time = 0
    estimated_distance = None

    client.start_session()
    start_time = time.time()

    try:
        while True:
            info, iq = client.get_next()
            amp = np.abs(iq)

            # Presence Detection
            if np.max(amp) > PRESENCE_AMP_THRESHOLD:
                if not present and (time.time() > presence_debounce_time):
                    present = True
                    print("Presence detected.")
                presence_debounce_time = time.time() + PRESENCE_DEBOUNCE_S
            elif present and (time.time() > presence_debounce_time):
                present = False
                estimated_distance = None
                phase_history.clear()
                time_history.clear()
                print("Presence lost.")
            
            if not present:
                time.sleep(1/UPDATE_RATE_HZ)
                continue

            # Distance Estimation
            if estimated_distance is None:
                distances = np.linspace(cfg.range_interval[0], cfg.range_interval[1], len(iq))
                estimated_distance = distances[np.argmax(amp)]
                print(f"Estimated distance to person: {estimated_distance:.2f} m")

            # Form Time Series
            distances = np.linspace(cfg.range_interval[0], cfg.range_interval[1], len(iq))
            dist_index = np.argmin(np.abs(distances - estimated_distance))
            phase = np.angle(iq[dist_index])
            phase_history.append(phase)
            time_history.append(time.time() - start_time)

            if len(phase_history) == time_series_len:
                # Bandpass Filter
                filtered_phase = bandpass_filter(np.unwrap(phase_history), UPDATE_RATE_HZ)

                # Estimate Breathing Rate
                power_spectrum = np.abs(np.fft.rfft(filtered_phase))**2
                freqs = np.fft.rfftfreq(len(filtered_phase), 1/UPDATE_RATE_HZ)
                breathing_rate_bpm = freqs * 60

                # Update Plots
                line1.set_data(time_history, filtered_phase)
                ax1.relim()
                ax1.autoscale_view()

                line2.set_data(breathing_rate_bpm, power_spectrum)
                ax2.relim()
                ax2.autoscale_view()
                ax2.set_xlim(BREATHING_RATE_RANGE_BPM)

                fig.canvas.draw()
                fig.canvas.flush_events()

                # Output estimated breathing rate
                valid_indices = (breathing_rate_bpm >= BREATHING_RATE_RANGE_BPM[0]) & (breathing_rate_bpm <= BREATHING_RATE_RANGE_BPM[1])
                if np.any(valid_indices):
                    peak_index = np.argmax(power_spectrum[valid_indices])
                    estimated_breathing_rate = breathing_rate_bpm[valid_indices][peak_index]
                    print(f"Estimated Breathing Rate: {estimated_breathing_rate:.2f} BPM")

            time.sleep(1/UPDATE_RATE_HZ)

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        client.stop_session()
        client.disconnect()
        print("\nDisconnected.")

if __name__ == "__main__":
    main()
