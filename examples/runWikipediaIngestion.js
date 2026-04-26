'use strict';

const { Brain } = require('../index');
const { ingestWikipediaArticle, evaluateQuestionSet } = require('../src/ingestion');

async function run() {
  const brain = new Brain({
    defaultTargetAccuracy: 0.7,
    epochsPerRound: 150,
    maxEpochsTotal: 3000,
    maxMutations: 6,
  });

  const articleTitle = 'Ada Lovelace';
  const result = await ingestWikipediaArticle(brain, articleTitle, {
    includeWikitext: true,
  });

  console.log(`Ingested ${result.lines.length} statements from ${articleTitle}.`);

  const evaluation = evaluateQuestionSet(brain, [
    { question: `Who is ${articleTitle}?`, expected: result.article.description },
    { question: `Where was ${articleTitle} born?`, expected: 'London' },
  ]);

  console.log(`QA accuracy: ${(evaluation.accuracy * 100).toFixed(1)}%`);
  evaluation.results.forEach(entry => {
    console.log(`Q: ${entry.question}`);
    console.log(`A: ${entry.value} (expected ${entry.expected})`);
  });
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
