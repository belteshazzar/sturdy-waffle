# sturdy-waffle — Digital Brain Framework

A JavaScript framework for a **digital brain** that starts with zero knowledge, detects unfamiliar problem domains, spawns specialised neural regions, and evolves through neuroplasticity until each lesson is learned. Brain state can be saved to and reloaded from JSON so that a trained brain never needs to relearn the same material.

---

## Table of Contents

1. [Concept](#concept)
2. [Brain Architecture](#brain-architecture)
   - [Neural Network](#neural-network)
   - [Brain Region](#brain-region)
   - [Router](#router)
   - [Knowledge Tree](#knowledge-tree)
   - [State Manager](#state-manager)
3. [Learning Lifecycle](#learning-lifecycle)
4. [Neuroplasticity & Mutation](#neuroplasticity--mutation)
5. [Getting Started](#getting-started)
6. [API Reference](#api-reference)
7. [Boolean Logic Syllabus](#boolean-logic-syllabus)
8. [Saving & Loading Brain State](#saving--loading-brain-state)
9. [Introspection](#introspection)
10. [Visualisation-Ready Design](#visualisation-ready-design)
11. [Project Structure](#project-structure)

---

## Concept

The digital brain is inspired by biological learning:

- **No prior knowledge** — a fresh Brain instance contains no regions and knows nothing.
- **Domain detection** — when a Lesson with an unknown domain is presented the Brain spawns a new BrainRegion sized relative to the input space.
- **Adaptive architecture** — if a region plateaus without reaching its accuracy target it *mutates* (adds neurons, adds layers, resets weights) and keeps training.
- **Neuroplasticity** — a newly spawned region is fully plastic (high learning rate). Once a lesson is learned the region consolidates: plasticity drops to ~0.1, making it stable and resistant to forgetting.
- **Compositional routing** — a hierarchical Router maps dot-notation domain keys to regions. Complex expressions are evaluated by recursively routing sub-expressions to the appropriate region (e.g. `AND(OR(a,b), NOT(c))`).
- **Persistence** — the complete brain state (all weights, biases, architectures, training history) serialises to JSON and can be reloaded instantly.

---

## Brain Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                               BRAIN                                     │
│                                                                         │
│   learnSyllabus(s) ──▶  learn(lesson) ──▶  _spawnRegion / retrain      │
│   evaluate(expr)   ──▶  Router.route(domain) ──▶ BrainRegion.predict   │
│                                                                         │
│  ┌──────────────────┐    ┌─────────────────────────────────────────┐   │
│  │     Router       │    │          Knowledge Regions               │   │
│  │                  │    │                                         │   │
│  │  * (root)        │    │  ┌─────────────┐  ┌─────────────┐      │   │
│  │  └─ boolean      │───▶│  │ boolean.AND │  │ boolean.XOR │ ...  │   │
│  │     ├─ AND  ─────│──▶ │  │             │  │             │      │   │
│  │     ├─ OR   ─────│──▶ │  │ NeuralNet   │  │ NeuralNet   │      │   │
│  │     ├─ NOT  ─────│──▶ │  │ [2, 14, 1]  │  │ [2, 8, 1]   │      │   │
│  │     ├─ XOR  ─────│──▶ │  │ acc: 100%   │  │ acc: 100%   │      │   │
│  │     ├─ NAND ─────│──▶ │  │ plastic:0.1 │  │ plastic:0.1 │      │   │
│  │     ├─ NOR  ─────│──▶ │  └─────────────┘  └─────────────┘      │   │
│  │     └─ XNOR ─────│──▶ │                                         │   │
│  └──────────────────┘    └─────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                       State Manager                              │  │
│  │   brain.save(path) ──▶ brain-state.json (weights, arch, meta)  │  │
│  │   Brain.load(path) ◀── brain-state.json                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Neural Network

`src/brain/NeuralNetwork.js`

A pure-JavaScript feedforward neural network with backpropagation. No external ML libraries are used.

| Property | Description |
|---|---|
| `architecture` | Array of layer sizes, e.g. `[2, 8, 1]` |
| `learningRate` | Gradient descent step size |
| `hiddenActivation` | Activation for hidden layers (`sigmoid`, `relu`, `tanh`, `linear`) |
| `outputActivation` | Activation for output layer |

**Key methods:**

| Method | Description |
|---|---|
| `predict(input)` | Forward pass → continuous output vector |
| `predictBinary(input, threshold)` | Forward pass → 0/1 output vector |
| `train(samples, epochs)` | Run backpropagation for N epochs |
| `accuracy(samples)` | Fraction of samples answered correctly |
| `addNeuronsToLayer(layerIdx, count)` | Grow a hidden layer in place |
| `addHiddenLayer(size, insertAt)` | Insert a new hidden layer |
| `reinitialize()` | Reset weights, keep architecture |
| `toJSON()` / `fromJSON(data)` | Serialise / deserialise |

**Backpropagation summary:**

```
Forward:  z[l] = W[l] · a[l−1] + b[l]
          a[l] = σ(z[l])

Output δ: δ[L] = (a[L] − y) ⊙ σ'(z[L])
Hidden δ: δ[l] = (W[l+1]ᵀ · δ[l+1]) ⊙ σ'(z[l])

Update:   W[l] −= lr · δ[l] · a[l−1]ᵀ
          b[l] −= lr · δ[l]
```

### Brain Region

`src/brain/BrainRegion.js`

A BrainRegion wraps a NeuralNetwork with lifecycle metadata:

| Property | Description |
|---|---|
| `domain` | Dot-notation domain key, e.g. `"boolean.AND"` |
| `plasticity` | 1.0 = fully malleable; 0.1 = consolidated |
| `accuracy` | Validation accuracy (0–1) |
| `trained` | `true` once target accuracy is met |
| `mutationCount` | Number of structural mutations applied |
| `trainingHistory` | Per-round snapshots of loss, accuracy, architecture |

**Training loop:**

```
while epochs < maxEpochsTotal:
    effectiveLR = baseLearningRate × plasticity
    train for epochsPerRound epochs
    measure accuracy on validation set
    if accuracy ≥ targetAccuracy → mark trained, consolidate, exit
    if accuracy plateau for mutationPatience rounds → mutate

Consolidation: plasticity = max(0.1, 1 − accuracy)
```

**Initial network sizing:**

The hidden layer size scales with the input space:

```javascript
hidden = clamp(inputSize × 6, 6, 32)
```

### Router

`src/routing/Router.js`

A trie-based (tree) router that maps domain strings to BrainRegions.

- Domains are split on `'.'` to build a hierarchy:
  `"boolean.AND"` → `root → boolean → AND`
- Routing is exact-match first, then prefix-walk (deepest node wins).
- Wildcard segments (`'*'`) are also supported.
- `getTreeStructure()` returns the full tree as a plain object for visualisation.

### Knowledge Tree

The Brain maintains a `knowledgeTree` object that mirrors the domain structure:

```json
{
  "boolean": {
    "AND":  { "_domain": "boolean.AND"  },
    "OR":   { "_domain": "boolean.OR"   },
    "NOT":  { "_domain": "boolean.NOT"  },
    "XOR":  { "_domain": "boolean.XOR"  },
    "NAND": { "_domain": "boolean.NAND" },
    "NOR":  { "_domain": "boolean.NOR"  },
    "XNOR": { "_domain": "boolean.XNOR" }
  }
}
```

This tree is included in introspection output and persisted to state files.

### State Manager

`src/persistence/StateManager.js`

Handles JSON serialisation/deserialisation of the entire Brain:

```
save(brain, path)    → writes brain.toJSON() to file (creates dirs)
load(path)           → reads JSON, reconstructs Brain with all regions
loadRaw(path)        → returns the raw JSON object for tooling
```

---

## Learning Lifecycle

```
1. Brain starts with no regions.

2. brain.learn(lesson)
   ├─ domain unknown?  → emit 'lesson:unknown'  → _spawnRegion(lesson)
   │                       • calculate initial hidden size
   │                       • new BrainRegion(domain, lesson)
   │                       • router.register(domain, region)
   │                       • emit 'region:spawned'
   └─ region.train()
       • training loop (see above)
       • emits 'training:progress' each round
       • emits 'mutation' when architecture changes
       • emits 'training:complete' when done
       → emit 'lesson:complete'

3. brain.learnSyllabus(syllabus)
   Iterates lessons in order, calling brain.learn() for each.

4. brain.evaluate(expression)
   Recursively resolves expression tree using the appropriate regions.
```

---

## Neuroplasticity & Mutation

**Plasticity** controls the effective learning rate:

```
effectiveLR = baseLearningRate × plasticity
```

A freshly spawned region has `plasticity = 1.0`, giving maximum LR.
After a lesson is learned, `plasticity = max(0.1, 1 − accuracy)`.
A perfectly accurate region has `plasticity ≈ 0.1` — 10× more stable.

When accuracy plateaus, one of four mutations is applied at random:

| Probability | Mutation | Effect |
|---|---|---|
| 35 % | Add neurons to hidden layer | Grows capacity of an existing layer |
| 25 % | Add hidden layer | Increases depth |
| 20 % | Boost learning rate | Helps escape shallow local minima |
| 20 % | Reinitialise weights | Full restart with current architecture |

Mutations continue until the target accuracy is reached or `maxMutations` is exhausted.

---

## Getting Started

### Prerequisites

- Node.js ≥ 18

### Install

```bash
git clone https://github.com/belteshazzar/sturdy-waffle.git
cd sturdy-waffle
npm install
```

### Run the boolean logic demo

```bash
npm start
# or
node examples/runBooleanLogic.js
```

Example output:

```
════════════════════════════════════════════════════════════
  Digital Brain Framework — Boolean Logic Demo
════════════════════════════════════════════════════════════

Brain created.  Regions: 0  (no knowledge yet)

Learning Boolean Logic Syllabus…
[Brain] Unknown domain 'boolean.AND' — spawning new region
[Brain] Region 'boolean.AND' spawned, arch=[2,12,1]
[Brain] Region 'boolean.AND' trained — acc=100.0%, mutations=0, epochs=300

  ✓ learned  boolean.AND    acc=100.0%  mutations=0  epochs=300
...
  ✓  AND(OR(1, 0), NOT(0)) = 1  (expected 1)
...
  Total: 18/18 tests passed
```

### Run tests

```bash
npm test
```

---

## API Reference

### `new Brain(config?)`

```javascript
const brain = new Brain({
  defaultTargetAccuracy: 0.99,   // accuracy to reach before consolidation
  maxMutations:          12,     // max structural mutations per region
  epochsPerRound:        500,    // training epochs between accuracy checks
  maxEpochsTotal:        30000,  // hard training budget per region
  verbose:               false,  // print progress to stdout
});
```

### `brain.learn(lesson)` → result

Present a single Lesson to the Brain. Spawns a new region if needed.

```javascript
const { trained, accuracy, mutationCount, totalEpochs } = brain.learn(myLesson);
```

### `brain.learnSyllabus(syllabus)` → results[]

Learn all lessons in a Syllabus in order.

### `brain.predict(input, domain)` → number[]

Raw (continuous) prediction from the region for `domain`.

### `brain.predictBinary(input, domain, threshold?)` → number[]

Thresholded (0/1) binary prediction.

### `brain.evaluate(expression)` → 0 | 1

Recursively evaluate a nested expression tree. Each non-leaf node maps to a domain via `"boolean.<OP>"`.

```javascript
// AND(OR(1, 0), NOT(0)) = 1
const result = brain.evaluate({
  op: 'AND', inputs: [
    { op: 'OR',  inputs: [{ value: 1 }, { value: 0 }] },
    { op: 'NOT', inputs: [{ value: 0 }]               },
  ],
});
```

### `brain.knows(domain)` → boolean

`true` if the region is trained and confirmed accurate.

### `brain.introspect()` → object

Returns a full snapshot (see [Introspection](#introspection)).

### `brain.save(filepath)` → { saved, filepath, sizeBytes, regionCount }

Persist brain state to a JSON file.

### `Brain.load(filepath)` → Brain

Restore a Brain from a saved state file.

---

### `new Lesson(opts)`

```javascript
const lesson = new Lesson({
  name:          'AND Gate',
  domain:        'boolean.AND',   // unique dot-notation key
  description:   'Boolean AND',
  trainingData:  [                 // { input: number[], output: number[] }
    { input: [0, 0], output: [0] },
    { input: [1, 1], output: [1] },
  ],
  validationData: [...],           // optional; defaults to trainingData
  tags:          ['boolean'],
});
```

### `new Syllabus(opts)`

```javascript
const syllabus = new Syllabus({
  name:        'My Syllabus',
  description: '...',
  lessons:     [lesson1, lesson2, ...],
});
syllabus.addLesson(lesson3);   // chainable
```

---

## Boolean Logic Syllabus

Located at `syllabi/booleanLogic/index.js`.

The syllabus teaches seven boolean gates in order of increasing difficulty:

| # | Domain | Gate | Truth Table | Notes |
|---|---|---|---|---|
| 1 | `boolean.AND`  | AND  | `00→0, 01→0, 10→0, 11→1` | Linearly separable |
| 2 | `boolean.OR`   | OR   | `00→0, 01→1, 10→1, 11→1` | Linearly separable |
| 3 | `boolean.NOT`  | NOT  | `0→1, 1→0`                | Single input |
| 4 | `boolean.XOR`  | XOR  | `00→0, 01→1, 10→1, 11→0` | Non-linearly separable — requires hidden layers |
| 5 | `boolean.NAND` | NAND | `00→1, 01→1, 10→1, 11→0` | Universal gate |
| 6 | `boolean.NOR`  | NOR  | `00→1, 01→0, 10→0, 11→0` | Universal gate |
| 7 | `boolean.XNOR` | XNOR | `00→1, 01→0, 10→0, 11→1` | Equivalence; hardest — often triggers mutations |

Usage:

```javascript
const Brain                    = require('./src/brain/Brain');
const { booleanLogicSyllabus } = require('./syllabi/booleanLogic');

const brain = new Brain({ verbose: true });
brain.learnSyllabus(booleanLogicSyllabus);

console.log(brain.evaluate({
  op: 'AND', inputs: [
    { op: 'OR',  inputs: [{ value: 1 }, { value: 0 }] },
    { op: 'NOT', inputs: [{ value: 1 }]               },
  ],
})); // → 0
```

---

## Saving & Loading Brain State

```javascript
const Brain = require('./src/brain/Brain');

// After learning:
brain.save('./saves/my-brain.json');

// In a future session (no retraining needed):
const brain = Brain.load('./saves/my-brain.json');
console.log(brain.evaluate({ op: 'AND', inputs: [{ value: 1 }, { value: 1 }] })); // → 1
```

The saved JSON includes:
- Brain version and creation timestamp
- Configuration
- Knowledge tree
- For each region: domain, trained flag, plasticity, accuracy, full neural network (all weights + biases), training history

---

## Introspection

```javascript
const info = brain.introspect();
```

Returns:

```json
{
  "version": "1.0.0",
  "createdAt": "2026-04-04T05:07:20.626Z",
  "regionCount": 7,
  "domains": ["boolean.AND", "boolean.OR", "boolean.NOT", "..."],
  "knowledgeTree": { "boolean": { "AND": { "_domain": "boolean.AND" }, "...": {} } },
  "routerTree": { "segment": "*", "children": { "boolean": { "...": {} } } },
  "regions": {
    "boolean.AND": {
      "domain": "boolean.AND",
      "trained": true,
      "plasticity": 0.1,
      "accuracy": 1.0,
      "mutationCount": 0,
      "architecture": [2, 12, 1],
      "trainingRounds": 1
    }
  }
}
```

---

## Visualisation-Ready Design

The framework is designed so that a visualisation or interactive UI layer can be added later without any changes to the core. Key design decisions that enable this:

### Event System

The Brain is an `EventEmitter`. Every significant lifecycle event is emitted with a structured payload:

| Event | Payload |
|---|---|
| `lesson:unknown` | `{ domain }` |
| `region:spawned` | `{ domain, architecture }` |
| `training:progress` | `{ domain, epoch, accuracy, loss, architecture, plasticity }` |
| `mutation` | `{ domain, mutationCount, type, architecture }` |
| `region:consolidated` | `{ domain, plasticity, accuracy }` |
| `region:trained` | `{ domain, trained, accuracy, mutationCount, totalEpochs, architecture }` |
| `lesson:complete` | `{ domain, lessonName, trained, accuracy, ... }` |
| `syllabus:start` | `{ name, lessonCount }` |
| `syllabus:complete` | `{ name, results }` |

Subscribe to any of these to drive real-time UI updates:

```javascript
brain.on('training:progress', ({ domain, epoch, accuracy, architecture }) => {
  ui.updateRegionProgress(domain, epoch, accuracy, architecture);
});

brain.on('mutation', ({ domain, type, architecture }) => {
  ui.showMutation(domain, type, architecture);
});
```

### Serialisable State

`brain.introspect()` returns a plain object that can be sent over a WebSocket or REST API. The routing tree, knowledge tree, and per-region metrics are all included.

### BrainRegion Training History

Each region stores `trainingHistory` — an array of per-round snapshots:

```json
[
  { "epoch": 500,  "loss": 0.21, "accuracy": 0.75, "architecture": [2, 12, 1], "plasticity": 1.0 },
  { "epoch": 1000, "loss": 0.08, "accuracy": 1.00, "architecture": [2, 12, 1], "plasticity": 1.0 }
]
```

This is ideal for plotting learning curves in a graph.

---

## Project Structure

```
sturdy-waffle/
├── index.js                          # Public API barrel export
├── package.json
├── README.md
│
├── src/
│   ├── brain/
│   │   ├── Brain.js                  # Top-level Brain container (EventEmitter)
│   │   ├── BrainRegion.js            # Specialised region with NN + plasticity
│   │   └── NeuralNetwork.js          # Feedforward net with backprop & mutation
│   ├── learning/
│   │   ├── Lesson.js                 # Unit of knowledge (domain + training data)
│   │   └── Syllabus.js               # Ordered collection of lessons
│   ├── routing/
│   │   └── Router.js                 # Hierarchical domain router (trie)
│   ├── persistence/
│   │   └── StateManager.js           # JSON save / load
│   └── utils/
│       ├── ActivationFunctions.js    # sigmoid, relu, tanh, linear + derivatives
│       └── MathUtils.js              # Matrix / vector helpers
│
├── syllabi/
│   └── booleanLogic/
│       └── index.js                  # 7-gate boolean logic syllabus
│
├── examples/
│   └── runBooleanLogic.js            # Full demo: train → test → save → load
│
├── tests/
│   ├── neuralNetwork.test.js         # NN unit tests
│   ├── brain.test.js                 # Brain unit tests
│   └── booleanLogic.test.js          # Integration: full syllabus + truth tables + nested
│
└── saves/                            # Runtime brain state files (gitignored)
```
