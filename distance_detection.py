
import time
import numpy as np
import acconeer.exptool as et
from collections import deque
import matplotlib.pyplot as plt

# ---- Tune these to your geometry ----
RANGE_INTERVAL = [0.2, 1.0]   # The range of distances to measure (in meters)
UPDATE_RATE_HZ = 10          # The update rate of the sensor

def main():
    args = et.a111.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    client = et.a111.Client(**et.a111.get_client_args(args))

    # IQ service config
    cfg = et.a111.IQServiceConfig()
    cfg.sensor = args.sensors
    cfg.range_interval = RANGE_INTERVAL
    cfg.update_rate = UPDATE_RATE_HZ
    
    client.connect()
    
    si = client.setup_session(cfg)
    print("Session info:", si)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Amplitude vs. Distance plot
    line1, = ax1.plot([], [])
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Live Radar Data")

    # Distance vs. Time plot
    time_history = deque(maxlen=int(5 * 60 * UPDATE_RATE_HZ)) # 5 minutes of history
    distance_history = deque(maxlen=int(5 * 60 * UPDATE_RATE_HZ))
    line2, = ax2.plot([], [])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Estimated Distance (m)")
    ax2.set_title("Estimated Distance Over Time")
    fig.tight_layout(pad=3.0)

    start_time = time.time()

    client.start_session()

    try:
        while True:
            info, iq = client.get_next()

            amp = np.abs(iq)
            num_bins = len(iq)
            distances = np.linspace(cfg.range_interval[0], cfg.range_interval[1], num_bins)
            
            # Estimate distance by finding the peak amplitude
            max_amp_index = np.argmax(amp)
            estimated_distance = distances[max_amp_index]

            # Update Amplitude vs. Distance plot
            line1.set_data(distances, amp)
            ax1.relim()
            ax1.autoscale_view()
            
            # Update Distance vs. Time plot
            current_time = time.time() - start_time
            time_history.append(current_time)
            distance_history.append(estimated_distance)

            line2.set_data(list(time_history), list(distance_history))
            ax2.relim()
            ax2.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()
            
            print(f"Estimated distance: {estimated_distance:.3f} m")

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
