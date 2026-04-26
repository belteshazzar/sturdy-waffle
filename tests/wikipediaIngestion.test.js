'use strict';

const Brain = require('../src/brain/Brain');
const {
  ingestWikipediaArticle,
  evaluateQuestionSet,
  normalizeEntity,
} = require('../src/ingestion');

function fastBrain(overrides = {}) {
  return new Brain({
    defaultTargetAccuracy: 0.7,
    epochsPerRound: 150,
    maxEpochsTotal: 3000,
    maxMutations: 6,
    ...overrides,
  });
}

describe('Wikipedia ingestion', () => {
  test('ingests statements and answers free-form questions', async () => {
    const brain = fastBrain();
    const article = {
      title: 'Ada Lovelace',
      normalizedTitle: 'Ada Lovelace',
      displayTitle: 'Ada Lovelace',
      description: 'English mathematician and writer',
      extract: 'Ada Lovelace was an English mathematician and writer. Ada Lovelace was born in London.',
      content_urls: {
        desktop: { page: 'https://en.wikipedia.org/wiki/Ada_Lovelace' },
      },
      wikitext: `{{Infobox person
| name = Ada Lovelace
| birth_date = 10 December 1815
| birth_place = London
}}`,
    };

    const result = await ingestWikipediaArticle(brain, 'Ada Lovelace', {
      article,
      retrain: true,
    });

    expect(result.trainedDomains.length).toBeGreaterThan(0);

    const subject = normalizeEntity('Ada Lovelace');
    expect(brain.queryAttribute(subject, 'birthPlace')).toBe('London');
    expect(brain.queryAttribute(subject, 'type')).toBe('English mathematician and writer');

    const answers = brain.answerFreeForm('Where was Ada Lovelace born?');
    expect(answers[0].value).toBe('London');

    const evaluation = evaluateQuestionSet(brain, [
      { question: 'Who is Ada Lovelace?', expected: 'English mathematician and writer' },
      { question: 'Where was Ada Lovelace born?', expected: 'London' },
    ]);
    expect(evaluation.accuracy).toBe(1);
  });
});
