# BeatNet

BeatNet finds beats and downbeats in your browser. Give it some audio, get the
timestamps, and keep the audio on your machine. Pretty neat.

This is a TypeScript port of [BeatNet](https://github.com/mjhydri/BeatNet) by
Mojtaba Heydari and collaborators. It runs their model 1 with ONNX Runtime Web,
including madmom's preprocessing and DBN beat tracking.

## Installation

```sh
npm install @jgorset/beatnet
```

Download [the model](https://github.com/jgorset/beatnet/releases/download/v1.0.0/beatnet.onnx)
and serve it from your application. It's about 1.6 MB. The preprocessing constants
and decoder state spaces are included in the package.

## Usage

```ts
import { createBeatNet } from '@jgorset/beatnet';

const tracker = await createBeatNet({ modelUrl: '/models/beatnet.onnx' });

try {
    // Mono audio sampled at 22,050 Hz.
    const audio = new Float32Array(/* your audio samples */);
    const { beats, downbeats } = await tracker.analyze(audio);

    console.log(beats);     // Seconds from the start of the audio.
    console.log(downbeats); // The first beat of each bar.
} finally {
    await tracker.dispose();
}
```

Keep the tracker around if you're analyzing more than one recording. Loading the
model for every song would be a little rude to your computer. Each analysis starts
with fresh recurrent state.

Decoding files, mixing channels and resampling audio are up to your application.
The tracker accepts mono PCM at exactly 22,050 Hz. Run it in a Web Worker to keep
your interface responsive. ONNX Runtime uses WebAssembly on the CPU.

### Long recordings

You don't have to keep the whole recording in memory. Supply a sample count and
a reader that returns exactly the requested samples:

```ts
const result = await tracker.analyze({
    sampleCount,
    read: async (first, count) => readAudioSamples(first, count),
}, {
    signal: abortController.signal,
    onProgress: (stage, completed, total) => console.log(stage, completed, total),
});
```

Reads are sequential and cover roughly 10 seconds plus context. Recurrent state
continues across windows. `beats` and `downbeats` contain seconds;
`beatConfidence` and `downbeatConfidence` contain probabilities at 20 ms
intervals. Beat confidence includes downbeats. The confidence arrays and decoder
working memory still grow with the recording; the decoder checkpoints its
traceback instead of storing every state transition.

Cancellation takes effect between windows and after an in-flight inference
finishes. Decoding is synchronous. To stop immediately, terminate the worker.
Wait for analysis to settle before disposing the tracker, and run one analysis
at a time per tracker.

### Serving WebAssembly

Your bundler needs to serve ONNX Runtime's WASM assets. With Vite, exclude
`onnxruntime-web` and `onnxruntime-web/webgpu` from dependency optimization so
those assets keep their paths. You can also serve the files yourself:

```ts
const tracker = await createBeatNet({
    modelUrl: '/models/beatnet.onnx',
    wasmPaths: '/onnxruntime/',
});
```

Copy the `ort-wasm-simd-threaded` WASM and module files from the pinned
`onnxruntime-web` package into that directory. The tracker uses one thread.
ONNX Runtime's WASM settings are shared within the page or worker, so set the
same paths before creating multiple trackers.

### Lower-level functions

`extractFeatures`, `inferBeatNet` and `decodeDbn` are exported if you want to
inspect features, work with an existing ONNX session or compare the decoder.
The preprocessing and decoder constants are pinned to madmom 0.16.1.

## Development

Use Node.js 24:

```sh
npm ci
npm test
```

The tests cover frozen preprocessing outputs, streamed context, recurrent state,
decoder traceback, cancellation, invalid data and tracker lifecycle. The port
was also compared with the implementation it was extracted from using the real
model: features, activations and decoded event times matched exactly across
multiple recurrent windows.

### Exporting the model

Use Python 3.10 or newer in a virtual environment:

```sh
python -m pip install -r scripts/requirements.txt
python -m pip install --no-deps BeatNet==1.1.1
python scripts/export.py --out .tmp/export
```

The exporter loads the upstream network and model 1 weights directly. It doesn't
need BeatNet's audio input or particle-filter dependencies. It checks the ONNX
model and prints its SHA-256 checksum; compare it with `model.json` before
replacing the released model. The export makes recurrent state explicit without
changing weights. Preprocessing and DBN constants stay pinned in `src`.

## License

The TypeScript implementation is MIT licensed. BeatNet's model and the adapted
exporter forward pass retain [CC BY 4.0](licenses/BeatNet-CC-BY-4.0.txt). madmom's
source algorithms retain [BSD-2-Clause](licenses/madmom-BSD-2-Clause.txt).
See [the attribution](THIRD_PARTY_NOTICES.md) before redistributing the model.

## Contribute

I love you.
