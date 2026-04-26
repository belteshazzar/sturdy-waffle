'use strict';

const {
  Brain,
  Lesson,
  Syllabus,
  FactBase,
} = require('..');

const originalRandom = Math.random;

beforeEach(() => {
  let seed = 42;
  Math.random = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
});

afterEach(() => {
  Math.random = originalRandom;
});

describe('advanced learning extensions', () => {
  test('records episodic + semantic memory and supports inference', () => {
    const brain = new Brain();
    const fb = new FactBase('Animals');
    fb.assert('bird', 'canFly', true)
      .assert('cat', 'canFly', false)
      .assertValue('bird', 'habitat', 'sky')
      .assertValue('cat', 'habitat', 'home');

    brain.learnFacts(fb);

    expect(brain.memory.episodic.episodes.length).toBeGreaterThan(0);
    expect(brain.memory.semantic.facts.length).toBeGreaterThan(0);

    const inference = brain.inferFact('bird', 'canFly');
    expect(inference.value).toBe(1);
  });

  test('shared embedding can be enabled for new regions', () => {
    const brain = new Brain({
      sharedEmbedding: { enabled: true, prototypeCount: 4 },
    });

    const lesson = new Lesson({
      name: 'Tiny AND',
      domain: 'tiny.AND',
      trainingData: [
        { input: [0, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ],
      inputSize: 2,
    });

    brain.learn(lesson);
    const region = brain.router.route('tiny.AND');
    expect(region.getInfo().embedding).toEqual(
      expect.objectContaining({ inputSize: 2 })
    );
  });

  test('baseline evaluation produces few-shot metrics', () => {
    const brain = new Brain();
    const lessonA = new Lesson({
      name: 'Tiny OR',
      domain: 'tiny.OR',
      trainingData: [
        { input: [0, 0], output: [0] },
        { input: [1, 0], output: [1] },
      ],
      inputSize: 2,
    });
    const lessonB = new Lesson({
      name: 'Tiny NOT',
      domain: 'tiny.NOT',
      trainingData: [
        { input: [0], output: [1] },
        { input: [1], output: [0] },
      ],
      inputSize: 1,
    });
    const syllabus = new Syllabus({
      name: 'Tiny syllabus',
      lessons: [lessonA, lessonB],
    });

    const report = brain.baselineReport({ syllabi: [syllabus], shots: 1 });
    expect(report.fewShot.length).toBe(2);
    expect(report.inventory.regionCount).toBe(0);
  });
});
