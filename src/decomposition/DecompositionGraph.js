'use strict';

const { TOKEN_NAMES } = require('./tokens');

/**
 * DecompositionGraph — records the incremental decomposition of a single
 * problem-solving episode as a directed acyclic graph (DAG).
 *
 * Neuroscience analogue: episodic memory trace stored in the hippocampus
 * during a problem-solving episode.  After the episode, the trace can be
 * replayed (see ReplayBuffer) to consolidate the controller's policy.
 *
 * Nodes represent either:
 *   • 'input'  — tokens from the original flat sequence
 *   • 'reduce' — the result of applying an operator to one or more operands
 *
 * Edges (implicit in each reduce-node's `children` field) represent
 * data-flow dependencies: operands feed into the operator output.
 *
 * The graph is built incrementally during a solve() episode:
 *   graph.init(tokens)     at episode start
 *   graph.addStep(...)     after each controller action
 */
class DecompositionGraph {
  constructor() {
    this.nodes       = [];   // all nodes in creation order
    this.steps       = [];   // ordered reduction steps
    this.inputTokens = [];   // original token sequence (for display)
    this._nextId     = 0;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /**
   * Initialise (or reset) the graph for a new problem.
   * @param {number[]} tokens  Flat token sequence for this episode
   */
  init(tokens) {
    this.nodes       = [];
    this.steps       = [];
    this.inputTokens = [...tokens];
    this._nextId     = 0;

    for (const tok of tokens) {
      const tokenId = (tok && typeof tok === 'object') ? tok.token : tok;
      const value = (tok && typeof tok === 'object') ? tok.value : null;
      this.nodes.push({
        id:       this._nextId++,
        type:     'input',
        token:    tokenId,
        value,
        children: [],
      });
    }
  }

  // ── Step recording ────────────────────────────────────────────────────────

  /**
   * Record a single reduction step.
   *
   * @param {number}   start   Working-memory index where the op token sat
   * @param {number}   op      Operation token integer
   * @param {number[]} args    Resolved operand values consumed
   * @param {number}   result  Value produced by this reduction
   * @returns {number}  ID of the newly created reduce-node
   */
  addStep(start, op, args, result) {
    const id   = this._nextId++;
    const node = {
      id,
      type:     'reduce',
      token:    op,
      value:    result,
      children: [...args],
    };
    this.nodes.push(node);
    this.steps.push({
      stepNo: this.steps.length,
      nodeId: id,
      start,
      op,
      opName: TOKEN_NAMES[op] ?? String(op),
      args:   [...args],
      result,
    });
    return id;
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  /** Plain-object summary safe for JSON serialisation. */
  toJSON() {
    return {
      nodeCount:   this.nodes.length,
      stepCount:   this.steps.length,
      inputTokens: this.inputTokens.map(t => {
        if (t && typeof t === 'object') {
          const name = TOKEN_NAMES[t.token] ?? t.token;
          return t.value !== undefined ? `${name}(${t.value})` : name;
        }
        return TOKEN_NAMES[t] ?? t;
      }),
      steps:       this.steps.map(s => ({
        stepNo: s.stepNo,
        op:     s.opName,
        args:   s.args,
        result: s.result,
      })),
    };
  }
}

module.exports = DecompositionGraph;
