'use strict';

const { TOKEN, ARITY } = require('../../src/decomposition/tokens');

// ── Helper: truth-table evaluation ───────────────────────────────────────────

/**
 * Evaluate a single Boolean operation deterministically from its truth table.
 * @param {number}   opTok  Operation token integer (TOKEN.AND etc.)
 * @param {number[]} args   Resolved operand values (each 0 or 1)
 * @returns {number} 0 or 1
 */
function evalOp(opTok, args) {
  switch (opTok) {
    case TOKEN.AND:  return args[0] & args[1];
    case TOKEN.OR:   return args[0] | args[1];
    case TOKEN.NOT:  return args[0] === 0 ? 1 : 0;
    case TOKEN.XOR:  return args[0] ^ args[1];
    case TOKEN.NAND: return (args[0] & args[1]) === 0 ? 1 : 0;
    case TOKEN.NOR:  return (args[0] | args[1]) === 0 ? 1 : 0;
    case TOKEN.XNOR: return (args[0] ^ args[1]) === 0 ? 1 : 0;
    default: throw new Error(`evalOp: unknown op token ${opTok}`);
  }
}

// ── Expression-tree utilities ─────────────────────────────────────────────────

/**
 * Build a random expression tree of the requested depth.
 *
 * At each non-leaf node one of `ops` is chosen at random.  At depth 0 a
 * random Boolean value (0 or 1) is always used.  To guarantee the expression
 * genuinely reaches the requested depth, the *first* child of every interior
 * node is always grown to `depth − 1`; other children are grown randomly
 * between 0 and `depth − 1`.
 *
 * @param {number}   depth   Maximum depth (0 = leaf)
 * @param {number[]} opToks  Operator token integers to sample from
 * @returns {{ type: 'value'|'op', value?: number, opTok?: number, children?: Array }}
 */
function randomTree(depth, opToks) {
  if (depth === 0) {
    // Use TOKEN.V0 / TOKEN.V1 explicitly so leaf values are always token integers
    return { type: 'value', value: Math.random() < 0.5 ? TOKEN.V0 : TOKEN.V1 };
  }
  const opTok = opToks[Math.floor(Math.random() * opToks.length)];
  const arity = ARITY[opTok];
  const children = [];
  for (let i = 0; i < arity; i++) {
    const childDepth = (i === 0)
      ? depth - 1
      : Math.floor(Math.random() * depth);
    children.push(randomTree(childDepth, opToks));
  }
  return { type: 'op', opTok, children };
}

/**
 * Convert an expression tree to a flat prefix-notation token array.
 * @param {{ type, value?, opTok?, children? }} tree
 * @returns {number[]}
 */
function treeToTokens(tree) {
  if (tree.type === 'value') return [tree.value];
  return [tree.opTok, ...tree.children.flatMap(c => treeToTokens(c))];
}

/**
 * Convert an expression tree to a human-readable prefix-notation string.
 * Operator names are taken from TOKEN_NAMES; value tokens become '0' or '1'.
 *
 * Example: AND(OR(1,0), NOT(0))
 *
 * @param {{ type, value?, opTok?, children? }} tree
 * @returns {string}
 */
function treeToString(tree) {
  const { TOKEN_NAMES } = require('../../src/decomposition/tokens');
  if (tree.type === 'value') return tree.value === 0 ? '0' : '1';
  const name = TOKEN_NAMES[tree.opTok] || String(tree.opTok);
  const args = tree.children.map(c => treeToString(c)).join(',');
  return `${name}(${args})`;
}

/**
 * Evaluate an expression tree against the Boolean truth tables.
 * @param {{ type, value?, opTok?, children? }} tree
 * @returns {number} 0 or 1
 */
function evalTree(tree) {
  if (tree.type === 'value') return tree.value;
  return evalOp(tree.opTok, tree.children.map(evalTree));
}

// ── Curriculum ────────────────────────────────────────────────────────────────

/**
 * DecompositionCurriculum — staged curriculum for training the
 * DecompositionController following a brain-like developmental trajectory.
 *
 * Neuroscience analogue: graduated practice from simple to complex tasks
 * mirrors the cortical maturation pathway — simple stimuli are mastered first,
 * building a foundation of "chunked" representations (depth-1 gates) that are
 * then recombined in increasingly deep hierarchies.
 *
 * Stage structure
 * ───────────────
 *  Stage 1 — depth 1: all single-gate truth-table examples (complete)
 *  Stage 2 — depth 2: one level of nesting, sampled across all gate pairs
 *  Stage 3 — depth 3: two levels of nesting, sampled more broadly
 *
 * Each stage returns an array of problems:
 *   { tokens: number[], answer: number }
 *
 * A matching evalFn is also exported so that Brain.trainDecomposition() can
 * supply it to computeExpertTrace() without needing to know about the
 * Boolean truth tables.
 */
class DecompositionCurriculum {
  /**
   * @param {object}   [opts]
   * @param {number[]} [opts.depth1Ops]  Op tokens for stage 1 (default: all 7)
   * @param {number[]} [opts.depth2Ops]  Op tokens for stage 2 (default: AND/OR/NOT/XOR)
   * @param {number[]} [opts.depth3Ops]  Op tokens for stage 3 (default: AND/OR/NOT/XOR)
   * @param {number}   [opts.depth2Count=80]  Sampled problems for stage 2
   * @param {number}   [opts.depth3Count=60]  Sampled problems for stage 3
   */
  constructor(opts = {}) {
    const ALL_OPS = [TOKEN.AND, TOKEN.OR, TOKEN.NOT, TOKEN.XOR, TOKEN.NAND, TOKEN.NOR, TOKEN.XNOR];

    this.depth1Ops   = opts.depth1Ops   || ALL_OPS;
    this.depth2Ops   = opts.depth2Ops   || [TOKEN.AND, TOKEN.OR, TOKEN.NOT, TOKEN.XOR];
    this.depth3Ops   = opts.depth3Ops   || [TOKEN.AND, TOKEN.OR, TOKEN.NOT, TOKEN.XOR];
    this.depth2Count = opts.depth2Count !== undefined ? opts.depth2Count : 80;
    this.depth3Count = opts.depth3Count !== undefined ? opts.depth3Count : 60;
  }

  // ── Stage generators ──────────────────────────────────────────────────────

  /**
   * Stage 1: exhaustive depth-1 problems — every op × every truth-table row.
   *
   * Each problem now also includes a `string` field with the human-readable
   * prefix-notation expression (e.g. "AND(1,0)") for use by StringEncoder.
   *
   * @returns {Array<{ tokens: number[], answer: number, string: string }>}
   */
  generateDepth1() {
    const { TOKEN_NAMES } = require('../../src/decomposition/tokens');
    const problems = [];
    for (const opTok of this.depth1Ops) {
      const arity   = ARITY[opTok];
      const opName  = TOKEN_NAMES[opTok];
      if (arity === 1) {
        for (const v of [0, 1]) {
          problems.push({
            tokens: [opTok, v],
            answer: evalOp(opTok, [v]),
            string: `${opName}(${v})`,
          });
        }
      } else {
        for (const a of [0, 1]) {
          for (const b of [0, 1]) {
            problems.push({
              tokens: [opTok, a, b],
              answer: evalOp(opTok, [a, b]),
              string: `${opName}(${a},${b})`,
            });
          }
        }
      }
    }
    return problems;
  }

  /**
   * Stage 2: sampled depth-2 problems (one outer op + at least one inner op).
   * @returns {Array<{ tokens: number[], answer: number, string: string }>}
   */
  generateDepth2() {
    return this._generateSampled(2, this.depth2Ops, this.depth2Count);
  }

  /**
   * Stage 3: sampled depth-3 problems (two levels of nesting).
   * @returns {Array<{ tokens: number[], answer: number, string: string }>}
   */
  generateDepth3() {
    return this._generateSampled(3, this.depth3Ops, this.depth3Count);
  }

  _generateSampled(depth, opToks, count) {
    const seen     = new Set();
    const problems = [];
    let   attempts = 0;
    const maxAttempts = count * 20;

    while (problems.length < count && attempts < maxAttempts) {
      attempts++;
      const tree   = randomTree(depth, opToks);
      const tokens = treeToTokens(tree);
      const key    = JSON.stringify(tokens);
      if (!seen.has(key)) {
        seen.add(key);
        problems.push({ tokens, answer: evalTree(tree), string: treeToString(tree) });
      }
    }
    return problems;
  }

  // ── Curriculum access ─────────────────────────────────────────────────────

  /**
   * Return all stages as an ordered array, each with a name, depth, and
   * problems array.
   * @returns {Array<{ name: string, depth: number, problems: Array }>}
   */
  getStages() {
    return [
      { name: 'Stage 1 — depth-1 (single gates)',  depth: 1, problems: this.generateDepth1() },
      { name: 'Stage 2 — depth-2 (one nesting)',   depth: 2, problems: this.generateDepth2() },
      { name: 'Stage 3 — depth-3 (two nestings)',  depth: 3, problems: this.generateDepth3() },
    ];
  }
}

// ── Singleton ─────────────────────────────────────────────────────────────────

const decompositionCurriculum = new DecompositionCurriculum();

module.exports = {
  DecompositionCurriculum,
  decompositionCurriculum,
  /** Exported so callers can pass it directly to computeExpertTrace(). */
  evalOp,
  /** Exported for testing and Phase 4 training data generation. */
  treeToString,
};
