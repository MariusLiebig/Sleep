import time
from xmlrpc import client
import numpy as np
import acconeer.exptool as et
# Robust import that works across exptool versions
try:
    # Newer exptool (3.x+)
    from acconeer.exptool.clients.base import ClientError
except ModuleNotFoundError:
    try:
        # Older exptool (2.x)
        from acconeer.exptool.a111 import ClientError
    except Exception:
        # Very old fallback: if ClientError isn't available, use a generic Exception
        class ClientError(Exception):
            pass
# put near the other imports, at the top
from acconeer.exptool.a111._clients.base import ClientError
from acconeer.exptool._core.communication.links.buffered_link import LinkError

from acconeer.exptool.a111.algo.sleep_breathing import (
    Processor as SBProcessor,
    ProcessingConfiguration as SBProcCfg,
)
from collections import deque
import statistics
import matplotlib.pyplot as plt

# Rolling windows
ROLL_SHORT_SEC = 60      # short-term variability
ROLL_LONG_SEC  = 5*60    # stage proxy window
PRINT_STATE_EVERY = 30   # seconds

# ---- Tune these to your geometry ----
RANGE_INTERVAL = [0.40, 1.0]   # make sure your sternum is inside
UPDATE_RATE_HZ = 10

# Gates / thresholds (tune!)
SNR_MIN = 2.0
BPM_SLEEP_MIN = 5.0
BPM_SLEEP_MAX = 24.0

AMP_AWAKE_RMS = 0.010    # radians (phase) RMS over short window considered "active" breathing/motion
BPM_STD_AWAKE = 3.0      # bpm std over 60s considered "awake-ish"

# Stage proxies (when asleep)
DEEP_BPM_MAX      = 12.0    # bpm
DEEP_BPM_STD_MAX  = 1.2     # bpm std over 5 min
REM_BPM_STD_MIN   = 2.0     # bpm std over 5 min suggests REM-ish

# Presence detection thresholds (tune to your setup)
PRESENCE_AMP_RMS_ON  = 0.006    # rad (phase RMS) to declare "present"
PRESENCE_AMP_RMS_OFF = 0.0045       # rad (phase RMS) to declare "not present"
PRESENCE_SNR_MIN     = 5.0      # minimum SNR to trust presence

# Debounce (seconds)
PRESENCE_ON_DEBOUNCE  = 5.0
PRESENCE_OFF_DEBOUNCE = 5.0


def safe_mean(x):
    try:
        return statistics.mean(x)
    except statistics.StatisticsError:
        return float('nan')


def safe_std(x):
    try:
        return statistics.pstdev(x)
    except statistics.StatisticsError:
        return float('nan')


def estimate_breath_amp(out):
    # Prefer phi_filt if present, else phi_raw
    phi = out.get("phi_filt", None)
    if phi is None:
        phi = out.get("phi_raw", None)
    if phi is None:
        return 0.0

    arr = np.asarray(phi, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(arr ** 2)))


def classify_state(bpm_mean, bpm_std_s, bpm_std_l, amp_rms, snr_mean):
    # Basic validity gate
    if not np.isfinite(bpm_mean) or not np.isfinite(bpm_std_s) or snr_mean < SNR_MIN:
        return "estimating"

    # Awake heuristics
    if amp_rms >= AMP_AWAKE_RMS:
        return "awake"
    if bpm_std_s >= BPM_STD_AWAKE:
        return "awake"
    if bpm_mean < BPM_SLEEP_MIN or bpm_mean > BPM_SLEEP_MAX:
        return "awake"

    # Sleep — stage proxy via variability + mean
    if bpm_mean <= DEEP_BPM_MAX and bpm_std_l <= DEEP_BPM_STD_MAX:
        return "sleep_deep"
    if bpm_std_l >= REM_BPM_STD_MIN:
        return "sleep_REM"
    return "sleep_light"


def auto_roi(client, cfg):
    """
    Compute a simple amplitude-based ROI from the currently running session.
    Assumes the IQ service session is already started.
    """
    print("Starting automatic ROI detection...")
    accum = None

    # allow a brief warm-up
    time.sleep(0.1)

    n_frames = int(5 * UPDATE_RATE_HZ)
    got = 0

    while got < n_frames:
        try:
            info, frame = client.get_next()  # IQ returns complex sweeps per bin
        except LinkError:
            # transient read hiccup; wait a bit and retry
            time.sleep(0.05)
            continue

        amp = np.abs(frame)  # amplitude proxy
        accum = np.maximum(accum, amp) if accum is not None else amp
        got += 1

    num_bins = accum.size
    r0, r1 = cfg.range_interval
    best_bin = int(np.argmax(accum))
    best_m = r0 + (r1 - r0) * (best_bin + 0.5) / num_bins
    roi_half = 0.08  # ±8 cm around best bin
    roi_lo = max(r0, best_m - roi_half)
    roi_hi = min(r1, best_m + roi_half)
    print(f"Auto ROI -> [{roi_lo:.2f}, {roi_hi:.2f}] m (best ~{best_m:.2f} m)")
    return (roi_lo, roi_hi)



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
    
    # Breathing processor config
    sb_cfg = SBProcCfg()
    sb_cfg.f_low = 0.02          # 1.2 BPM
    sb_cfg.f_high = 0.6         # 36 BPM
    sb_cfg.n_dft = 12.0         # shorter warm-up while testing
    sb_cfg.t_freq_est = 0.25
    sb_cfg.lambda_p = 60.0
    sb_cfg.lambda_05 = 1.0

    # Data buffers
    bpm_short = deque(maxlen=int(ROLL_SHORT_SEC * UPDATE_RATE_HZ))
    bpm_long  = deque(maxlen=int(ROLL_LONG_SEC * UPDATE_RATE_HZ))
    amp_short = deque(maxlen=int(ROLL_SHORT_SEC * UPDATE_RATE_HZ))
    snr_short = deque(maxlen=int(ROLL_SHORT_SEC * UPDATE_RATE_HZ))

    present = False
    state_change_deadline = 0.0
    last_state_print = 0.0
    
    proc = None

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Amplitude vs. Distance plot
    line1, = ax1.plot([], [])
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Live Radar Data")

    # Distance vs. Time plot
    time_history = deque(maxlen=int(ROLL_LONG_SEC * UPDATE_RATE_HZ))
    distance_history = deque(maxlen=int(ROLL_LONG_SEC * UPDATE_RATE_HZ))
    line2, = ax2.plot([], [])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Chest Distance (m)")
    ax2.set_title("Chest Distance Over Time")
    fig.tight_layout(pad=3.0)

    start_time = time.time()

    try:
        si = client.setup_session(cfg)
        print("Session info:", si)
        client.start_session()

        while True:
            info, iq = client.get_next()

            amp = np.abs(iq)
            num_bins = len(iq)
            distance = np.linspace(cfg.range_interval[0], cfg.range_interval[1], num_bins)
            line1.set_data(distance, amp)
            ax1.relim()
            ax1.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if proc:
                out = proc.process(iq, info)
            else:
                out = None

            if out is not None:
                chest_dist = out.get("dist_est", float('nan'))
                if np.isfinite(chest_dist):
                    current_time = time.time() - start_time
                    time_history.append(current_time)
                    distance_history.append(chest_dist)

                    line2.set_data(list(time_history), list(distance_history))
                    ax2.relim()
                    ax2.autoscale_view()

            if out is None:    
                # Still process presence to start the sensor
                amp_rms_raw = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
                
                now = time.time()
                want_present = (amp_rms_raw >= PRESENCE_AMP_RMS_ON)

                if not present and want_present:
                    if state_change_deadline == 0.0:
                        state_change_deadline = now + PRESENCE_ON_DEBOUNCE
                    elif now >= state_change_deadline:
                        present = True
                        state_change_deadline = 0.0
                        print(">> Presence detected: START measuring")
                        
                        roi = auto_roi(client, cfg)
                        cfg.range_interval = roi

                        # To apply a changed service config, stop → setup → start once
                        try:
                            client.stop_session()
                        except ClientError:
                            pass

                        si = client.setup_session(cfg)
                        proc = SBProcessor(cfg, sb_cfg, si)
                        client.start_session()

                        # Clear rolling buffers for a clean start after ROI change
                        bpm_short.clear(); bpm_long.clear()
                        amp_short.clear(); snr_short.clear()

                        # Reset rolling buffers for clean start
                        bpm_short.clear(); bpm_long.clear()
                        amp_short.clear(); snr_short.clear()
                elif not present:
                     state_change_deadline = 0.0
                     print(f"[Idle] amp_rms_raw={amp_rms_raw:.4f} rad")
                     time.sleep(0.1)

                continue

            ip = out.get("init_progress", 0)
            if ip is not None and ip < 100:
                if time.time() - last_state_print > 0.5:
                    print(f"Init {ip}%")
                    last_state_print = time.time()
                continue

            # Extract current estimates
            f_est = float(out.get("f_est", 0.0))
            snr = float(out.get("snr", 0.0))
            bpm = f_est * 60.0 if f_est > 0 else 0.0
            bpm_dft = float(out.get("f_dft_est", 0.0))*60
            amp_rms = estimate_breath_amp(out)

            # -------- Presence detection with debounce --------
            now = time.time()
            want_absent  = (snr <  PRESENCE_SNR_MIN) or  (amp_rms <= PRESENCE_AMP_RMS_OFF)

            if present and want_absent:
                if state_change_deadline == 0.0:
                    state_change_deadline = now + PRESENCE_OFF_DEBOUNCE
                elif now >= state_change_deadline:
                    present = False
                    proc = None  # Stop processing
                    state_change_deadline = 0.0
                    print("<< Presence lost: STOP measuring")
                    try:
                        client.stop_session()
                    except ClientError:
                        pass
                    si = client.setup_session(cfg)
                    client.start_session()

            elif present:
                state_change_deadline = 0.0


            if not present:
                continue

            # -------- Present: update metrics & classify --------
            if np.isfinite(bpm):
                bpm_short.append(bpm)
                bpm_long.append(bpm)
            snr_short.append(snr)
            amp_short.append(amp_rms)

            print(f"BPM: {bpm:5.2f} BMP DFT: {bpm_dft:5.2f} (SNR={snr:.2f}, Amp_RMS={amp_rms:.4f})")

            # Every PRINT_STATE_EVERY seconds, output state
            if now - last_state_print >= PRINT_STATE_EVERY:
                last_state_print = now
                bpm_mean_s = safe_mean(bpm_short)
                bpm_std_s  = safe_std(bpm_short)
                bpm_mean_l = safe_mean(bpm_long)
                bpm_std_l  = safe_std(bpm_long)
                snr_mean   = safe_mean(snr_short)
                amp_mean   = safe_mean(amp_short)

                state = classify_state(bpm_mean_l, bpm_std_s, bpm_std_l, amp_mean, snr_mean)

                print(
                    f"\n[State] {state} | "
                    f"bpm_mean_1m={bpm_mean_s:.2f}, bpm_std_1m={bpm_std_s:.2f}, "
                    f"bpm_mean_5m={bpm_mean_l:.2f}, bpm_std_5m={bpm_std_l:.2f}, "
                    f"amp_rms={amp_mean:.4f} rad, snr={snr_mean:.2f}\n"
                )

    except KeyboardInterrupt:
        pass
    finally:
        try:
            client.stop_session()
        except ClientError:
            pass
        client.disconnect()


if __name__ == "__main__":
    main()