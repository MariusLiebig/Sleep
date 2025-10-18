import time
import numpy as np
import acconeer.exptool as et
from acconeer.exptool.a111.algo.sleep_breathing import (
    Processor as SBProcessor,
    ProcessingConfiguration as SBProcCfg,
)

# ---- Tune these to your geometry ----
RANGE_INTERVAL = [0.40, 1.0]   # make sure your sternum is inside
UPDATE_RATE_HZ = 10

def main():
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

    except KeyboardInterrupt:
        pass
    finally:
        client.stop_session()
        client.disconnect()

if __name__ == "__main__":
    main()
