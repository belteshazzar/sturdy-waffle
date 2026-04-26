'use strict';

const WorkingMemory           = require('./WorkingMemory');
const DecompositionGraph      = require('./DecompositionGraph');
const DecompositionController = require('./DecompositionController');
const ReplayBuffer            = require('./ReplayBuffer');
const tokens                  = require('./tokens');

const { TOKEN, ARITY } = tokens;

/**
 * Compute the expert decomposition trace for a flat token sequence.
 *
 * The expert policy is "leftmost-first": at each step it applies the valid
 * reduction with the smallest start index.  This corresponds to a left-to-
 * right, bottom-up (post-order) evaluation of the prefix expression — the
 * most natural sequential decomposition of a tree.
 *
 * The returned trace is a sequence of (stateVec, chosenStart, op, args, result)
 * tuples that can be fed directly to DecompositionController.trainImitation().
 *
 * Neuroscience analogue: generating an "expert demonstration" for imitation
 * learning is analogous to observational learning / mirror-neuron activation —
 * the controller watches correct behaviour and updates its policy accordingly
 * before any trial-and-error exploration begins.
 *
 * @param {number[]} inputTokens  Flat prefix-notation token array
 * @param {function(number, number[]): number} evalFn
 *   Callback that evaluates a single operation: (opToken, argValues) → result.
 *   The caller (Brain or curriculum) supplies this so that the trace generator
 *   does not need to know about specialist BrainRegions.  For curriculum
 *   generation a deterministic truth-table function is used.
 * @param {number} [maxSlots=16]
 *
 * @returns {{
 *   trace:  Array<{ stateVec: number[], chosenStart: number, op: number,
 *                   args: number[], result: number }>,
 *   solved: boolean,
 *   answer: number | null
 * }}
 */
function computeExpertTrace(inputTokens, evalFn, maxSlots = 16) {
  const mem   = new WorkingMemory(maxSlots);
  mem.load(inputTokens);

  const trace = [];
  let   steps = 0;
  const maxSteps = maxSlots * 2;   // generous upper bound

  while (!mem.isSolved() && steps < maxSteps) {
    const candidates = mem.validReductions();
    if (candidates.length === 0) break;

    // Expert: always pick the leftmost valid reduction
    candidates.sort((a, b) => a.start - b.start);
    const action = candidates[0];

    const stateVec    = mem.toVector();
    const rawSlots    = [...mem.slots];          // full maxSlots-length raw token array
    const rawValues   = [...mem.values];
    const validStarts = candidates.map(c => c.start);  // all valid positions at this state
    const result      = evalFn(action.op, action.args);

    trace.push({
      stateVec,
      rawSlots,
      rawValues,
      validStarts,
      chosenStart: action.start,
      op:          action.op,
      args:        [...action.args],
      result,
    });

    mem.reduce(action.start, ARITY[action.op], result);
    steps++;
  }

  return {
    trace,
    solved: mem.isSolved(),
    answer: mem.answer(),
  };
}

module.exports = {
  WorkingMemory,
  DecompositionGraph,
  DecompositionController,
  ReplayBuffer,
  EmbeddingTable:   require('./EmbeddingTable'),
  GatingNetwork:    require('./GatingNetwork'),
  LearnedRouter:    require('./LearnedRouter'),
  StringEncoder:    require('./StringEncoder'),
  VQCodebook:       require('./VQCodebook'),
  tokens,
  computeExpertTrace,
};
