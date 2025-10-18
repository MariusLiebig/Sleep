import time
import numpy as np
import acconeer.exptool as et
from acconeer.exptool.a111.algo.sleep_breathing import (
    Processor as SBProcessor,
    ProcessingConfiguration as SBProcCfg,
)
from collections import deque
import statistics

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

# --- Add just before start of main loop ---
bpm_short = deque(maxlen=ROLL_SHORT_SEC)      # assume ~1 estimate per second
bpm_long  = deque(maxlen=ROLL_LONG_SEC)
amp_short = deque(maxlen=ROLL_SHORT_SEC)
snr_short = deque(maxlen=ROLL_SHORT_SEC)

def main():
    last_state_print = 0.0

    args = et.a111.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    client = et.a111.Client(**et.a111.get_client_args(args))

    # >>> IQ service (NOT Envelope) <<<
    cfg = et.a111.IQServiceConfig()
    cfg.sensor = args.sensors
    cfg.range_interval = RANGE_INTERVAL
    cfg.update_rate = UPDATE_RATE_HZ
    # Optional but often good for respiration:
    # cfg.downsampling_factor = 1
    # cfg.gain = 0.5

    client.connect()
    si = client.setup_session(cfg)
    print("Session info:", si)

    # --- quick 5s pre-scan to auto-pick best bin for ROI ---
    client.start_session()
    accum = None
    for _ in range(int(5 * UPDATE_RATE_HZ)):
        _, frame = client.get_next()  # IQ returns complex sweeps per bin
        amp = np.abs(frame)           # use amplitude as a simple proxy here
        accum = np.maximum(accum, amp) if accum is not None else amp
    num_bins = accum.size
    r0, r1 = cfg.range_interval
    best_bin = int(np.argmax(accum))
    best_m = r0 + (r1 - r0) * (best_bin + 0.5) / num_bins
    roi_half = 0.08  # ±8 cm around best bin
    roi_lo = max(r0, best_m - roi_half)
    roi_hi = min(r1, best_m + roi_half)
    print(f"Auto ROI -> [{roi_lo:.2f}, {roi_hi:.2f}] m (best ~{best_m:.2f} m)")
    client.stop_session()

    # --- breathing processor config ---
    sb_cfg = SBProcCfg()
    sb_cfg.f_low = 0.02          # 6 BPM
    sb_cfg.f_high = 0.6         # 36 BPM
    sb_cfg.n_dft = 12.0         # shorter warm-up while testing
    sb_cfg.t_freq_est = 0.25
    sb_cfg.lambda_p = 60.0
    sb_cfg.lambda_05 = 1.0
    # sb_cfg.dist_range = (roi_lo, roi_hi)  # **important**

    # Re-setup & start with same cfg
    si = client.setup_session(cfg)
    proc = SBProcessor(cfg, sb_cfg, si)
    client.start_session()

    t0 = time.time()
    try:
        # seen_keys = False
        # while True:
        #     info, iq = client.get_next()  # complex IQ per bin (shape: (bins,))
        #     out = proc.process(iq, info)

        #     if out is None:
        #         print(f"Initializing… {(time.time()-t0):.1f}s")
        #         continue

        #     if not seen_keys:
        #         print("Result keys:", list(out.keys()))
        #         seen_keys = True

        #     ip = out.get("init_progress")
        #     if ip is not None and ip < 100:
        #         print(f"Init {ip}%")
        #         continue

        last_print = 0
        SNR_MIN = 0.0  # tune; start with 1–3

        while True:
            info, iq = client.get_next()
            out = proc.process(iq, info)

            if out is None:
                continue

            ip = out.get("init_progress", 0)
            if ip is not None:
                if ip < 100:
                    if time.time() - last_print > 0.5:
                        print(f"Init {ip}%")
                        last_print = time.time()
                    continue
            # print(out.keys())
            f_dft_est = float(out.get("f_dft_est"))     # Hz
            f_est     = float(out.get("f_est"))         # Hz
            snr   = float(out.get("snr", 0.0))

            # print(f_est)
            # print(float(out.get("f_dft_est")))
            if (f_est > 0 and snr >= SNR_MIN) or f_dft_est > 0:
                bpm = f_est * 60.0
                raw_bpm = f_dft_est * 60.0
                print(f"BPM: {bpm:5.2f} , raw BPM: {raw_bpm:5.2f} (f_est={f_est:.3f} Hz, SNR={snr:.2f})")
            else:
                # helpful debug
                print(f"No reliable estimate yet (f_est={f_est:.3f} Hz, SNR={snr:.2f})")

            # inside while True: after you compute bpm, raw_bpm, snr...
            amp_rms = estimate_breath_amp(out)

            # Update rolling buffers (1 Hz loop assumed)
            if np.isfinite(bpm):
                bpm_short.append(bpm)
                bpm_long.append(bpm)
            snr_short.append(snr)
            amp_short.append(amp_rms)

            # Print continuous BPM line as you already do
            if (f_est > 0 and snr >= 0) or f_dft_est > 0:
                print(f"BPM: {bpm:5.2f} , raw BPM: {raw_bpm:5.2f} (f_est={f_est:.3f} Hz, SNR={snr:.2f})")
            else:
                print(f"No reliable estimate yet (f_est={f_est:.3f} Hz, SNR={snr:.2f})")

            # Every PRINT_STATE_EVERY seconds, output state
            now = time.time()
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
                    f"[State] {state} | "
                    f"bpm_mean_1m={bpm_mean_s:.2f}, bpm_std_1m={bpm_std_s:.2f}, "
                    f"bpm_mean_5m={bpm_mean_l:.2f}, bpm_std_5m={bpm_std_l:.2f}, "
                    f"amp_rms={amp_mean:.4f} rad, snr={snr_mean:.2f}"
                )


    except KeyboardInterrupt:
        pass
    finally:
        client.stop_session()
        client.disconnect()

if __name__ == "__main__":
    main()
