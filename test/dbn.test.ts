import assert from 'node:assert/strict';
import { test } from 'node:test';
import { decodeDbn } from '../dist/index.js';

test('DBN handles silence and reconstructs a path across traceback checkpoints', () => {
    const config = { threshold: .05, observationLambda: 16, hmms: [{
        states: [1, 0], pointers: [0, 1, 2], logTransitions: [0, 0],
        observations: [2, 0], positions: [0, .5], initial: [Math.log(.5), Math.log(.5)],
    }] };
    assert.deepEqual(decodeDbn(new Float32Array(1200), config).beats, []);
    const scores = new Float32Array(1202);
    for (let i = 0; i < 601; i++) { scores[i * 2] = .01; scores[i * 2 + 1] = i % 2 ? .01 : .9; }
    const result = decodeDbn(scores, config), expected = Array.from({ length: 301 }, (_, i) => i * 2 / 50);
    assert.deepEqual(result.beats, expected); assert.deepEqual(result.downbeats, expected);
    assert.ok(result.workingBytes > 0);
});

test('decoder rejects invalid probabilities and supports the packaged state spaces', () => {
    assert.deepEqual(decodeDbn(new Float32Array(100)).beats, []);
    for (const scores of [Float32Array.of(0), Float32Array.of(NaN, 0), Float32Array.of(-.1, .2), Float32Array.of(.8, .8)]) {
        assert.throws(() => decodeDbn(scores), /probabilities/);
    }
});
