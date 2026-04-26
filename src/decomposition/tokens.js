'use strict';

/**
 * Token vocabulary for the decomposition engine.
 *
 * Flat integer tokens represent both resolved values (V0/V1) and unresolved
 * operation placeholders (AND, OR, …) inside a WorkingMemory slot.  NULL (-1)
 * marks an empty slot and is encoded as an all-zeros one-hot vector.
 *
 * Neuroscience analogue: tokens are the "symbols" manipulated in prefrontal
 * working memory — a compact, discretised representation of the current
 * problem state that the controller (PFC) can attend to and act upon.
 */

// ── Token vocabulary ──────────────────────────────────────────────────────────

const TOKEN = Object.freeze({
  NULL:  -1,  // empty / padding slot
  V0:     0,  // resolved boolean false
  V1:     1,  // resolved boolean true
  AND:    2,
  OR:     3,
  NOT:    4,
  XOR:    5,
  NAND:   6,
  NOR:    7,
  XNOR:   8,
  VALUE:  9,  // resolved numeric value (non-binary)
  ADD:    10,
  SUB:    11,
  MUL:    12,
  DIV:    13,
  SQRT:   14,
  IMP:    15,
});

/** Reverse mapping: integer → token name string */
const TOKEN_NAMES = Object.fromEntries(
  Object.entries(TOKEN).map(([k, v]) => [v, k])
);

// ── Operator arity ────────────────────────────────────────────────────────────

/** Number of operands each operator consumes */
const ARITY = Object.freeze({
  [TOKEN.AND]:  2,
  [TOKEN.OR]:   2,
  [TOKEN.XOR]:  2,
  [TOKEN.NAND]: 2,
  [TOKEN.NOR]:  2,
  [TOKEN.XNOR]: 2,
  [TOKEN.NOT]:  1,
  [TOKEN.ADD]:  2,
  [TOKEN.SUB]:  2,
  [TOKEN.MUL]:  2,
  [TOKEN.DIV]:  2,
  [TOKEN.SQRT]: 1,
  [TOKEN.IMP]:  2,
});

/** All operator tokens (excludes NULL, V0, V1) */
const OPERATIONS = Object.freeze([
  TOKEN.AND, TOKEN.OR, TOKEN.NOT,
  TOKEN.XOR, TOKEN.NAND, TOKEN.NOR, TOKEN.XNOR,
  TOKEN.ADD, TOKEN.SUB, TOKEN.MUL, TOKEN.DIV, TOKEN.SQRT, TOKEN.IMP,
]);

/** Value tokens (resolved booleans) */
const VALUES = Object.freeze([TOKEN.V0, TOKEN.V1, TOKEN.VALUE]);

/**
 * One-hot vocabulary size: tokens 0..15 (V0 through IMP).
 * NULL (-1) is always encoded as an all-zeros slot — it is NOT included in the
 * vocab to avoid any index collisions.
 */
const VOCAB_SIZE = 16;  // V0=0 … IMP=15

/** Domain hints for shared symbol vocabulary (op name → domain list) */
const OP_DOMAIN_HINTS = Object.freeze({
  AND:  ['boolean.AND', 'fuzzy.AND'],
  OR:   ['boolean.OR', 'fuzzy.OR'],
  NOT:  ['boolean.NOT', 'fuzzy.NOT'],
  XOR:  ['boolean.XOR', 'fuzzy.XOR'],
  NAND: ['boolean.NAND'],
  NOR:  ['boolean.NOR'],
  XNOR: ['boolean.XNOR'],
  IMP:  ['fuzzy.IMP'],
  ADD:  ['math.ADD'],
  SUB:  ['math.SUB'],
  MUL:  ['math.MUL'],
  DIV:  ['math.DIV'],
  SQRT: ['math.SQRT'],
});

module.exports = {
  TOKEN,
  TOKEN_NAMES,
  ARITY,
  OPERATIONS,
  VALUES,
  VOCAB_SIZE,
  OP_DOMAIN_HINTS,
};
