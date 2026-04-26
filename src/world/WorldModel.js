'use strict';

/**
 * WorldModel captures simple transition statistics between states for
 * predictive coding and next-state expectations.
 */
class WorldModel {
  constructor({ maxTransitions = 5000 } = {}) {
    this.maxTransitions = maxTransitions;
    this.transitions = new Map(); // stateKey -> { count, sum, last }
  }

  _key(state, action, context) {
    const stateKey = state.map(v => v.toFixed(4)).join('|');
    const actionKey = action ? action.map(v => v.toFixed(4)).join('|') : '';
    const contextKey = context ? context.map(v => v.toFixed(4)).join('|') : '';
    return `${stateKey}::${actionKey}::${contextKey}`;
  }

  observe(state, nextState, opts = {}) {
    const action = Array.isArray(opts) ? opts : opts.action;
    const context = opts && !Array.isArray(opts) ? opts.context : null;
    const key = this._key(state, action, context);
    if (!this.transitions.has(key)) {
      this.transitions.set(key, { count: 0, sum: new Array(nextState.length).fill(0), last: null });
    }
    const entry = this.transitions.get(key);
    entry.count++;
    for (let i = 0; i < nextState.length; i++) {
      entry.sum[i] += nextState[i];
    }
    entry.last = nextState;

    if (this.transitions.size > this.maxTransitions) {
      const firstKey = this.transitions.keys().next().value;
      this.transitions.delete(firstKey);
    }
  }

  predict(state, opts = {}) {
    const action = Array.isArray(opts) ? opts : opts.action;
    const context = opts && !Array.isArray(opts) ? opts.context : null;
    const entry = this.transitions.get(this._key(state, action, context));
    if (!entry || entry.count === 0) return null;
    return entry.sum.map(v => v / entry.count);
  }

  rollout(initialState, { actions = [], context = null, steps = null } = {}) {
    const horizon = steps ?? actions.length;
    const states = [];
    let current = [...initialState];
    for (let i = 0; i < horizon; i++) {
      const action = actions[i] || null;
      const next = this.predict(current, { action, context });
      if (!next) break;
      states.push(next);
      current = next;
    }
    return states;
  }

  getInfo() {
    return {
      transitionCount: this.transitions.size,
      maxTransitions:  this.maxTransitions,
    };
  }

  toJSON() {
    return {
      maxTransitions: this.maxTransitions,
      transitions:    [...this.transitions.entries()],
    };
  }

  static fromJSON(data) {
    const model = new WorldModel({ maxTransitions: data.maxTransitions });
    model.transitions = new Map(data.transitions || []);
    return model;
  }
}

module.exports = WorldModel;
