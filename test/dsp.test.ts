import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { extractFeatures } from '../dist/index.js';
import { magnitudeTransform } from '../dist/dsp.js';

const audio = (length: number) => Float32Array.from({ length }, (_, i) =>
    .17 * Math.sin(i * .019) + .08 * Math.cos(i * .043) + (i % 11025 < 60 ? .6 : 0));

test('production preprocessing retains frozen outputs at recording and recurrent-window boundaries', () => {
    for (const [samples, expected] of [
        [1, 'b5f20ab795e9d5ff62d0792cd8e8575338a948bce5bf87ac4afde97ab2eabec9'],
        [441, '9e50fc769a6b5851e0dfd663668799a3fda8157561abe7571124643ff3930e59'],
        [22050, '288a39d20453f87b56833675aaf143ded7d637cd84cde6eaba0afb534bfdfe4a'],
        [441001, '8767fced9033ccd69000df400369bb8a8971403c59c64954e8a3fd923cd68207'],
    ] as const) {
        const values = extractFeatures(audio(samples));
        assert.equal(values.length, Math.ceil(samples / 441) * 272);
        assert.equal(createHash('sha256').update(Buffer.from(values.buffer)).digest('hex'), expected);
    }
});

test('streamed feature windows retain FFT context and positive differences', () => {
    const pcm = audio(441001), full = extractFeatures(pcm), frames = full.length / 272;
    const joined = new Float32Array(full.length);
    for (let first = 0; first < frames; first += 500) {
        const count = Math.min(500, frames - first);
        const begin = Math.max(0, (first - 1) * 441 - 1411), end = Math.min(pcm.length, (first + count) * 441 + 1411);
        joined.set(extractFeatures(pcm.subarray(begin, end), first, count, pcm.length, begin), first * 272);
    }
    assert.deepEqual(joined, full);
});

test('Bluestein FFT matches a direct DFT and clears reusable buffers', () => {
    const size = 17, pcm = Float64Array.from({ length: size }, (_, i) => Math.sin(i * .4));
    const transform = magnitudeTransform(size), actual = new Float32Array(size);
    transform(pcm, actual);
    for (let bin = 0; bin < size; bin++) {
        let real = 0, imaginary = 0;
        for (let i = 0; i < size; i++) { const a = 2 * Math.PI * bin * i / size; real += pcm[i] * Math.cos(a); imaginary -= pcm[i] * Math.sin(a); }
        assert.ok(Math.abs(actual[bin] - Math.hypot(real, imaginary)) < .000001);
    }
    transform(new Float64Array(size), actual);
    assert.ok(actual.every(value => value === 0));
});

test('preprocessing rejects corrupt samples, bounds and incomplete context', () => {
    assert.throws(() => extractFeatures(new Float32Array()), /finite mono/);
    assert.throws(() => extractFeatures(Float32Array.of(NaN)), /finite mono/);
    assert.throws(() => extractFeatures(audio(1000), -1), /bounds/);
    assert.throws(() => extractFeatures(audio(1000), 0, 4), /bounds/);
    assert.throws(() => extractFeatures(audio(441), 500, 1, 441001, 500 * 441), /context/);
});
