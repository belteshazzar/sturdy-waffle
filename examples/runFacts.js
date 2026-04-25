'use strict';

/**
 * runFacts.js — Declarative Knowledge & Simple Reasoning Demo
 *
 * This example shows how the Brain learns a small set of animal-kingdom facts
 * and then uses them to perform simple reasoning by combining fact lookups with
 * boolean logic operators inside Brain.evaluate() expression trees.
 *
 * Run:
 *   node examples/runFacts.js
 */

const Brain  = require('../src/brain/Brain');
const Lesson = require('../src/learning/Lesson');
const { animalFacts, animalSyllabus } = require('../syllabi/facts');

// ── 1. Set up a Brain ────────────────────────────────────────────────────────

const brain = new Brain({
  defaultTargetAccuracy: 0.95,
  epochsPerRound:        500,
  maxEpochsTotal:        20000,
  maxMutations:          10,
  verbose:               false,
});

console.log('=== Declarative Knowledge & Simple Reasoning ===\n');
console.log(`Teaching Brain: "${animalSyllabus.name}"`);
console.log(`  Subjects   : ${animalFacts.subjects.join(', ')}`);
console.log(`  Predicates : ${animalFacts.predicates.join(', ')}`);
console.log(`  Facts      : ${animalFacts.factCount} ground-truth assertions\n`);

// ── 2. Learn the facts ───────────────────────────────────────────────────────

brain.learnFacts(animalFacts);

console.log('Trained regions:');
const info = brain.introspect();
for (const domain of info.domains) {
  const r = info.regions[domain];
  console.log(`  ${domain.padEnd(28)} acc=${(r.accuracy * 100).toFixed(1)}%`);
}
console.log();

// ── 3. Direct fact queries ───────────────────────────────────────────────────

console.log('── Direct fact queries ──────────────────────────────');
const queries = [
  ['bird',    'canFly'],
  ['bat',     'canFly'],
  ['fish',    'canFly'],
  ['penguin', 'canFly'],
  ['cat',     'hasFur'],
  ['fish',    'hasFur'],
  ['frog',    'canSwim'],
  ['cat',     'canSwim'],
  ['bird',    'isWarmBlooded'],
  ['fish',    'isWarmBlooded'],
];
for (const [subject, predicate] of queries) {
  const predicted = brain.queryFact(subject, predicate);
  const expected  = animalFacts.get(subject, predicate);
  const correct   = predicted === expected ? '✓' : '✗';
  console.log(
    `  ${correct} queryFact('${subject}', '${predicate}')` +
    `  → ${predicted}  (expected ${expected})`
  );
}
console.log();

// ── 4. Simple reasoning with boolean logic ───────────────────────────────────

// We need AND and NOT to compose compound queries.
brain.learn(new Lesson({
  name:   'AND Gate',
  domain: 'boolean.AND',
  trainingData: [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [0] },
    { input: [1, 0], output: [0] },
    { input: [1, 1], output: [1] },
  ],
}));

brain.learn(new Lesson({
  name:   'NOT Gate',
  domain: 'boolean.NOT',
  trainingData: [
    { input: [0], output: [1] },
    { input: [1], output: [0] },
  ],
}));

console.log('── Reasoning: fact lookups + boolean operators ──────');

/**
 * Helper to print a reasoning query with its result.
 */
function reason(label, expression) {
  const result = brain.evaluate(expression);
  console.log(`  ${label.padEnd(50)} → ${result}`);
}

// "Does bat fly AND have fur?"  (bat can fly AND has fur → expected 1)
reason(
  'AND( canFly(bat), hasFur(bat) )',
  {
    op: 'AND',
    inputs: [
      { fact: { subject: 'bat', predicate: 'canFly'  } },
      { fact: { subject: 'bat', predicate: 'hasFur'  } },
    ],
  }
);

// "Can penguin swim but NOT fly?"  (penguin canSwim=T, canFly=F → expected 1)
reason(
  'AND( canSwim(penguin), NOT(canFly(penguin)) )',
  {
    op: 'AND',
    inputs: [
      { fact: { subject: 'penguin', predicate: 'canSwim' } },
      {
        op: 'NOT',
        inputs: [{ fact: { subject: 'penguin', predicate: 'canFly' } }],
      },
    ],
  }
);

// "Is frog warm-blooded AND can it fly?"  (frog: isWarmBlooded=F, canFly=F → expected 0)
reason(
  'AND( isWarmBlooded(frog), canFly(frog) )',
  {
    op: 'AND',
    inputs: [
      { fact: { subject: 'frog', predicate: 'isWarmBlooded' } },
      { fact: { subject: 'frog', predicate: 'canFly'        } },
    ],
  }
);

// "Is fish NOT warm-blooded?"  (fish: isWarmBlooded=F → NOT(0)=1 → expected 1)
reason(
  'NOT( isWarmBlooded(fish) )',
  {
    op: 'NOT',
    inputs: [{ fact: { subject: 'fish', predicate: 'isWarmBlooded' } }],
  }
);

// "Can bird fly AND is it NOT warm-blooded?"  (bird: canFly=T, isWarmBlooded=T → NOT(T)=0 → AND=0)
reason(
  'AND( canFly(bird), NOT(isWarmBlooded(bird)) )',
  {
    op: 'AND',
    inputs: [
      { fact: { subject: 'bird', predicate: 'canFly' } },
      {
        op: 'NOT',
        inputs: [{ fact: { subject: 'bird', predicate: 'isWarmBlooded' } }],
      },
    ],
  }
);

console.log('\nDone.');
