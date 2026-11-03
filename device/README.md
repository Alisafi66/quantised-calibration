# On-device benchmark

Measures inference latency for each precision on an Android device through
ONNX Runtime, with conditions interleaved so that thermal drift affects them
equally.

## Building

The ONNX Runtime Android library is fetched from Maven rather than committed,
since it is 17 MB and versioned independently of this project.

```bash
V=1.20.0
curl -sLo /tmp/ort.aar \
  "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${V}/onnxruntime-android-${V}.aar"
unzip -oq /tmp/ort.aar -d /tmp/ortaar
mkdir -p device/ort/lib && cp -r /tmp/ortaar/headers device/ort/include
cp /tmp/ortaar/jni/arm64-v8a/libonnxruntime.so device/ort/lib/

NDK=$HOME/Library/Android/sdk/ndk/28.2.13676358
$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android29-clang++ \
  -std=c++17 -O2 -fPIE -pie -static-libstdc++ \
  -Idevice/ort/include -Ldevice/ort/lib -lonnxruntime \
  -Wl,-rpath,'$ORIGIN' -o device/bench device/bench.cpp
```

`-static-libstdc++` is required: a stock Android image has no `libc++_shared.so`
and the binary will not link on the device without it.

## Running

Enable USB debugging on the device, connect it, authorise the host, then:

```bash
./device/run_benchmark.sh models/balanced_seed0 results/device/latency.json
```

The script pushes the binary, the runtime and the three graphs, records the
battery temperature either side of the run, and writes the results with the
device profile attached.

Latency depends on graph structure rather than on weight values, so one seed is
sufficient.
