#!/usr/bin/env bash
# Measure inference latency for each precision on the Android device.
#
# Conditions are interleaved inside one process so that thermal drift affects
# all of them equally, and the device temperature is recorded either side of
# the run so that a drifting measurement can be identified rather than
# silently averaged in.
set -euo pipefail

ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
REMOTE=/data/local/tmp/qcal
MODELS="${1:-models/balanced_seed0}"
OUT="${2:-results/device/latency.json}"

temp() { "$ADB" shell dumpsys battery 2>/dev/null | grep temperature | tr -d '\r ' | cut -d: -f2; }

"$ADB" shell "mkdir -p $REMOTE"
"$ADB" push -q device/bench "$REMOTE/" >/dev/null
"$ADB" push -q device/ort/lib/libonnxruntime.so "$REMOTE/" >/dev/null
"$ADB" shell "chmod 755 $REMOTE/bench"
for m in fp32 int8 int4; do "$ADB" push -q "$MODELS/$m.onnx" "$REMOTE/" >/dev/null; done

START=$(temp)
echo "device temperature before: $(echo "scale=1; $START/10" | bc) C"
RESULT=$("$ADB" shell "cd $REMOTE && LD_LIBRARY_PATH=. ./bench \
    --iters 200 --warmup 20 --rounds 5 --batch 1 --threads 1 \
    fp32=fp32.onnx int8=int8.onnx int4=int4.onnx" | tr -d '\r')
END=$(temp)
echo "device temperature after:  $(echo "scale=1; $END/10" | bc) C"

mkdir -p "$(dirname "$OUT")"
python3 - "$OUT" "$START" "$END" <<PY
import json, sys, subprocess
out, start, end = sys.argv[1], int(sys.argv[2]) / 10, int(sys.argv[3]) / 10
data = json.loads('''$RESULT''')
data["device"] = {
    "model": "Samsung Galaxy Tab S7 FE (SM-T733)",
    "soc": "Qualcomm SM7325",
    "android": "14 (API 34)",
    "power": "mains, USB connected",
    "temperature_c_before": start,
    "temperature_c_after": end,
}
open(out, "w").write(json.dumps(data, indent=2) + "\n")
print("written", out)
PY
