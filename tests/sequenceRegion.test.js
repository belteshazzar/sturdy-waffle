'use strict';

const Lesson = require('../src/learning/Lesson');
const SequenceBrainRegion = require('../src/brain/SequenceBrainRegion');

describe('SequenceBrainRegion', () => {
  test('trains on simple last-token task', () => {
    const lesson = new Lesson({
      name: 'Last Token',
      domain: 'sequence.LAST_TOKEN',
      trainingData: [
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [0] },
        { input: [0, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ],
      inputSize: 1,
      outputSize: 1,
      mode: 'classification',
      sequence: true,
    });

    const region = new SequenceBrainRegion({
      domain: lesson.domain,
      lesson,
      config: { targetAccuracy: 0.5, epochsPerRound: 100, maxEpochsTotal: 800 },
    });

    const result = region.train();
    expect(result.trained).toBe(true);
    const output = region.predict([1, 0])[0];
    expect(Number.isNaN(output)).toBe(false);
  });
});
