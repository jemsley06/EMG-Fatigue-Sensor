// MyoWare 2.0 RAW -> ESP32 ADC1 -> serial lines "t_us,raw" at FS_HZ.
// Record on the laptop with: python -m tools.capture <port> <out.csv>
//
// Wiring: MyoWare powered from 3.3 V (at 5 V its output can exceed the ESP32's 3.3 V pin limit),
// RAW output -> EMG_PIN, grounds shared. Laptop on battery while electrodes are on a person.

const int EMG_PIN = 34;                      // ADC1 only: ADC2 fails while Wi-Fi is on
const int64_t FS_HZ = 2000;                  // 1 kHz aliases through MyoWare's gentle 498 Hz low-pass
const int64_t PERIOD_US = 1000000 / FS_HZ;

void setup() {
  Serial.begin(921600);                      // ~32 kB/s of text at 2 kHz; 115200 baud would drop samples
  analogReadResolution(12);                  // 0-4095; default 11 dB attenuation covers ~0-3.1 V
}

void loop() {
  // TODO(you): sample EMG_PIN every PERIOD_US microseconds and print one "t_us,raw" line.
  // Use esp_timer_get_time() (64-bit microseconds, never wraps) for the schedule and the timestamp.
  // - Schedule from the last *scheduled* time or from *now*? One drifts over a 5-minute recording.
  // - Busy-wait here, not a timer ISR: analogRead() and Serial are not safe inside an ESP32 ISR.
  // - Print the actual read time, not the scheduled time, so load_myoware can detect jitter/gaps.
}
