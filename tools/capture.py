"""Record MyoWare + ESP32 serial output to a `t_us,raw` CSV. Stop with Ctrl-C.

    python -m tools.capture /dev/cu.usbserial-0001 recordings/biceps_01.csv
"""
import argparse

import serial


def main():
    ap = argparse.ArgumentParser(description="Record ESP32 EMG serial output to CSV.")
    ap.add_argument("port", help="serial port, e.g. /dev/cu.usbserial-0001 (ls /dev/cu.*)")
    ap.add_argument("out", help="output CSV path")
    ap.add_argument("--baud", type=int, default=921600)
    args = ap.parse_args()

    n = bad = 0
    with serial.Serial(args.port, args.baud, timeout=1) as ser, open(args.out, "w") as f:
        ser.reset_input_buffer()   # drop boot messages and stale bytes
        ser.readline()             # first line is likely cut off mid-number
        f.write("t_us,raw\n")
        print("recording... Ctrl-C to stop")
        try:
            while True:
                parts = ser.readline().strip().split(b",")
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    f.write(f"{int(parts[0])},{int(parts[1])}\n")
                    n += 1
                elif parts != [b""]:   # empty = read timeout, not corruption
                    bad += 1
        except KeyboardInterrupt:
            pass
    print(f"wrote {n} samples to {args.out} ({bad} malformed lines skipped)")


if __name__ == "__main__":
    main()
