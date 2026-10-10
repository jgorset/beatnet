import assert from 'node:assert/strict';
import { mock, test } from 'node:test';
import * as ort from 'onnxruntime-web/webgpu';
import { createBeatNet } from '../dist/index.js';

test('tracker returns confidence, resets each recording and releases its session once', async () => {
    let released = 0, runs = 0;
    mock.method(ort.InferenceSession, 'create', async (_, options) => {
        assert.deepEqual(options.executionProviders, ['wasm']);
        return { async run({ features, hidden, cell }) {
            assert.equal(hidden.data[0], 0); assert.equal(cell.data[0], 0); runs++;
            const count = features.dims[1], values = new Float32Array(count * 3);
            for (let i = 0; i < count; i++) values.set([.01, .01, .98], i * 3);
            return { probabilities: new ort.Tensor('float32', values, [1, count, 3]),
                hiddenOut: new ort.Tensor('float32', new Float32Array(300), [2, 1, 150]), cellOut: new ort.Tensor('float32', new Float32Array(300), [2, 1, 150]) };
        }, async release() { released++; } };
    });
    try {
        const tracker = await createBeatNet({ modelUrl: 'model.onnx' }), progress = [];
        const result = await tracker.analyze(new Float32Array(22050), { onProgress: stage => progress.push(stage) });
        assert.deepEqual(result.beats, []); assert.deepEqual(result.downbeats, []);
        assert.equal(result.beatConfidence.length, 50); assert.equal(result.beatConfidence[0], Math.fround(.01) * 2);
        assert.equal(result.downbeatConfidence[0], Math.fround(.01));
        assert.deepEqual(progress, ['BeatNet inference', 'Decoding beats']);
        await tracker.analyze(new Float32Array(441)); assert.equal(runs, 2);
        await tracker.dispose(); await tracker.dispose(); assert.equal(released, 1);
        await assert.rejects(tracker.analyze(new Float32Array(1)), /disposed/);
    } finally { mock.restoreAll(); }
});

test('failed and cancelled reads allow retry, while overlapping analysis and disposal are rejected', async () => {
    mock.method(ort.InferenceSession, 'create', async () => ({ async release() {} }));
    try {
        const tracker = await createBeatNet({ modelUrl: 'model.onnx' });
        await assert.rejects(tracker.analyze(new Float32Array()), /nonempty/);
        await assert.rejects(tracker.analyze({ sampleCount: 1, read: async () => Float32Array.of(NaN) }), /finite mono/);
        await assert.rejects(tracker.analyze({ sampleCount: 22050, read: async () => Float32Array.of(0) }), /audio window/);
        let resolve;
        const controller = new AbortController();
        const first = tracker.analyze({ sampleCount: 22050, read: () => new Promise(r => { resolve = r; }) }, { signal: controller.signal });
        await assert.rejects(tracker.analyze(new Float32Array(1)), /already analyzing/);
        await assert.rejects(tracker.dispose(), /finish/);
        controller.abort(); resolve(new Float32Array(22050));
        await assert.rejects(first, { name: 'AbortError' });
        await assert.rejects(tracker.analyze({ sampleCount: 22050, read: async () => { throw new Error('retry reached reader'); } }), /retry reached reader/);
        await tracker.dispose();
    } finally { mock.restoreAll(); }
});
