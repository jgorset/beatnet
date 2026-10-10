import assert from 'node:assert/strict';
import { mock, test } from 'node:test';
import * as ort from 'onnxruntime-web/webgpu';
import { inferBeatNet } from '../dist/index.js';

function output(count: number, marker = 1) {
    const values = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) values.set([.2, .1, .7], i * 3);
    return { probabilities: new ort.Tensor('float32', values, [1, count, 3]),
        hiddenOut: new ort.Tensor('float32', new Float32Array(300).fill(marker), [2, 1, 150]),
        cellOut: new ort.Tensor('float32', new Float32Array(300).fill(marker * 2), [2, 1, 150]) };
}

test('recurrent windows cover every frame and carry both states without leaking tensors', async () => {
    const disposed = new Set();
    const original = ort.Tensor.prototype.dispose;
    mock.method(ort.Tensor.prototype, 'dispose', function () { assert.ok(!disposed.has(this)); disposed.add(this); original.call(this); });
    try {
        for (const frames of [1, 499, 500, 501, 1001]) {
            let calls = 0, written = 0;
            const session = { async run({ features, hidden, cell }) {
                assert.equal(hidden.data[0], calls); assert.equal(cell.data[0], calls * 2);
                const count = features.dims[1]; assert.equal(features.data[0], written);
                calls++; written += count; return output(count, calls);
            } };
            const previous = disposed.size;
            const result = await inferBeatNet(session, frames, async (first, count) => { const values = new Float32Array(count * 272); values[0] = first; return values; });
            assert.equal(written, frames); assert.equal(calls, Math.ceil(frames / 500));
            assert.equal(result.length, frames * 2); assert.equal(result[0], Math.fround(.2));
            assert.equal(disposed.size - previous, 2 + calls * 4);
        }
    } finally { mock.restoreAll(); }
});

test('cancellation and malformed model outputs release every temporary tensor', async () => {
    let disposed = 0;
    const original = ort.Tensor.prototype.dispose;
    mock.method(ort.Tensor.prototype, 'dispose', function () { disposed++; original.call(this); });
    try {
        const controller = new AbortController();
        await assert.rejects(inferBeatNet({ async run({ features }) { controller.abort(); return output(features.dims[1]); } }, 501,
            async (_, count) => new Float32Array(count * 272), undefined, controller.signal), { name: 'AbortError' });
        assert.equal(disposed, 6);
        const start = disposed;
        await assert.rejects(inferBeatNet({ async run({ features }) { const values = output(features.dims[1]); values.probabilities.data[0] = NaN; return values; } }, 10,
            async (_, count) => new Float32Array(count * 272)), /probability output/);
        assert.equal(disposed - start, 6);
    } finally { mock.restoreAll(); }
});

test('failed model and feature reads surface as errors', async () => {
    await assert.rejects(inferBeatNet({ run: async () => { throw new Error('model failed'); } }, 10, async (_, count) => new Float32Array(count * 272)), /model failed/);
    await assert.rejects(inferBeatNet({}, 10, async () => new Float32Array()), /feature window/);
    await assert.rejects(inferBeatNet({}, 10, async () => { throw new Error('read failed'); }), /read failed/);
    await assert.rejects(inferBeatNet({}, 0, async () => new Float32Array()), /frame count/);
});
