'use strict';

const { TOKEN, ARITY, VOCAB_SIZE } = require('./tokens');

/**
 * WorkingMemory — short-term workspace for incremental problem decomposition.
 *
 * Neuroscience analogue: the dorsolateral prefrontal cortex maintains a
 * working-memory buffer of currently active representations.  Here each slot
 * can hold either a *resolved* value (0 or 1) or an *unresolved* operation
 * token (AND, OR, …).
 *
 * The memory is initialised from a flat prefix-notation token sequence and is
 * iteratively *reduced*: whenever an operation token is immediately followed
 * by enough resolved operands, the DecompositionController can choose to
 * collapse that segment into a single resolved value (a "reduction step").
 * The process repeats until a single resolved value remains — the answer.
 *
 * Example trajectory for AND(OR(1,0), NOT(0)):
 *   Load:  [AND, OR, 1, 0, NOT, 0]
 *   Step 1 – reduce OR at 1:  [AND, 1, NOT, 0]
 *   Step 2 – reduce NOT at 2: [AND, 1, 1]
 *   Step 3 – reduce AND at 0: [1]            → solved
 */
class WorkingMemory {
  /**
   * @param {number} [maxSlots=16]  Maximum number of slots.  Expressions
   *   longer than this value will be silently truncated at load() time.
   *   A depth-3 Boolean expression needs at most ~15 tokens.
   */
  constructor(maxSlots = 16) {
    this.maxSlots = maxSlots;
    this.slots    = new Array(maxSlots).fill(TOKEN.NULL);
    this.length   = 0;   // number of active (non-NULL) slots
  }

  // ── Loading ───────────────────────────────────────────────────────────────

  /**
   * Initialise the buffer from a flat array of token integers.
   * Any slots beyond `maxSlots` are silently dropped.
   * @param {number[]} tokens
   */
  load(tokens) {
    this.slots  = new Array(this.maxSlots).fill(TOKEN.NULL);
    const n     = Math.min(tokens.length, this.maxSlots);
    for (let i = 0; i < n; i++) this.slots[i] = tokens[i];
    this.length = n;
  }

  // ── Reduction ─────────────────────────────────────────────────────────────

  /**
   * Apply a single reduction at position `start`.
   *
   * Replaces the `1 + arity` contiguous slots beginning at `start`
   * (op token followed by its operands) with the single `result` value.
   * Trailing NULL padding is extended so the buffer remains exactly
   * `maxSlots` entries long.
   *
   * @param {number} start   Index of the operation token in `this.slots`
   * @param {number} arity   Number of operands consumed (from ARITY[op])
   * @param {number} result  Resolved value (0 or 1) to write back
   */
  reduce(start, arity, result) {
    const consumed = 1 + arity;           // op slot + operand slots
    this.slots.splice(start, consumed, result);
    // Restore the buffer to exactly maxSlots by appending NULLs
    for (let k = 0; k < arity; k++) this.slots.push(TOKEN.NULL);
    this.slots = this.slots.slice(0, this.maxSlots);
    this.length -= arity;                  // net change: removed arity, added 0
  }

  // ── Inspection ────────────────────────────────────────────────────────────

  /**
   * True when the memory has been fully reduced to a single resolved value.
   * @returns {boolean}
   */
  isSolved() {
    return this.length === 1 &&
           (this.slots[0] === TOKEN.V0 || this.slots[0] === TOKEN.V1);
  }

  /**
   * Return the final answer (0 or 1), or null if not yet solved.
   * @returns {number|null}
   */
  answer() {
    return this.isSolved() ? this.slots[0] : null;
  }

  /**
   * Enumerate all currently valid reduction positions.
   *
   * A reduction at index `i` is valid when:
   *   • `slots[i]` is a known operator token (present in ARITY)
   *   • the immediately following `arity(slots[i])` slots all hold resolved
   *     values (0 or 1)
   *
   * The basal-ganglia gating mechanism in DecompositionController uses this
   * list to filter its candidate set before selecting an action.
   *
   * @returns {Array<{ start: number, op: number, args: number[] }>}
   */
  validReductions() {
    const reductions = [];
    for (let i = 0; i < this.length; i++) {
      const tok = this.slots[i];
      if (!(tok in ARITY)) continue;   // not an operator

      const a = ARITY[tok];
      if (i + a >= this.maxSlots) continue;   // would exceed buffer

      const args = [];
      let ok = true;
      for (let j = 1; j <= a; j++) {
        const v = this.slots[i + j];
        if (v !== TOKEN.V0 && v !== TOKEN.V1) { ok = false; break; }
        args.push(v);
      }
      if (ok) reductions.push({ start: i, op: tok, args });
    }
    return reductions;
  }

  // ── Neural encoding ───────────────────────────────────────────────────────

  /**
   * Encode the current slot state as a flat float32 feature vector suitable
   * for input to the DecompositionController's neural network.
   *
   * Each of the `maxSlots` slots is one-hot encoded over the `VOCAB_SIZE`
   * token classes (indices 0..8 correspond to V0..XNOR).  A NULL slot is
   * encoded as an all-zeros segment.
   *
   * Total length: maxSlots × VOCAB_SIZE
   *
   * @returns {number[]}
   */
  toVector() {
    const vec = new Array(this.maxSlots * VOCAB_SIZE).fill(0);
    for (let i = 0; i < this.maxSlots; i++) {
      const tok = this.slots[i];
      if (tok >= 0 && tok < VOCAB_SIZE) {
        vec[i * VOCAB_SIZE + tok] = 1;
      }
      // NULL (tok === -1) → all-zeros segment (already 0)
    }
    return vec;
  }

  // ── Utilities ─────────────────────────────────────────────────────────────

  /** Return an independent copy of this WorkingMemory. */
  clone() {
    const m   = new WorkingMemory(this.maxSlots);
    m.slots   = [...this.slots];
    m.length  = this.length;
    return m;
  }

  /** Human-readable slot listing (active slots only). */
  toString() {
    const { TOKEN_NAMES } = require('./tokens');
    const names = this.slots.slice(0, this.length).map(t => {
      if (t === TOKEN.NULL) return 'NULL';
      if (t === TOKEN.V0)   return '0';
      if (t === TOKEN.V1)   return '1';
      return TOKEN_NAMES[t] || String(t);
    });
    return `[${names.join(', ')}]`;
  }
}

module.exports = WorkingMemory;
