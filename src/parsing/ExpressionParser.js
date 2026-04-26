'use strict';

const { TOKEN, OP_DOMAIN_HINTS } = require('../decomposition/tokens');

/**
 * ExpressionParser — parse human-readable expressions into structured trees
 * and token streams compatible with Brain.evaluate() and the decomposition engine.
 */
class ExpressionParser {
  static tokenize(input, opts = {}) {
    const { maxTokens = 512, maxLength = 20000 } = opts;
    if (typeof input !== 'string') {
      throw new Error('ExpressionParser: input must be a string');
    }
    if (input.length > maxLength) {
      throw new Error(`ExpressionParser: input exceeds max length (${maxLength})`);
    }
    const tokens = [];
    const pushToken = (token) => {
      tokens.push(token);
      if (tokens.length > maxTokens) {
        throw new Error(`ExpressionParser: too many tokens (>${maxTokens})`);
      }
    };
    let i = 0;
    const src = input.trim();
    const isIdentStart = ch => /[A-Za-z_]/.test(ch);
    const isIdentChar = ch => /[A-Za-z0-9_.]/.test(ch);
    while (i < src.length) {
      const ch = src[i];
      if (/\s/.test(ch)) { i++; continue; }
      if (ch === '(' || ch === ')' || ch === ',') {
        pushToken({ type: ch, value: ch });
        i++;
        continue;
      }
      if (ch === '"' || ch === '\'') {
        const quote = ch;
        let j = i + 1;
        let value = '';
        while (j < src.length && src[j] !== quote) {
          value += src[j];
          j++;
        }
        if (j >= src.length) throw new Error('ExpressionParser: unterminated string literal');
        pushToken({ type: 'string', value });
        i = j + 1;
        continue;
      }
      if (ch === '-' || /[0-9.]/.test(ch)) {
        let j = i;
        let seenDot = false;
        if (src[j] === '-') j++;
        while (j < src.length) {
          const cj = src[j];
          if (cj === '.') {
            if (seenDot) break;
            seenDot = true;
            j++;
            continue;
          }
          if (!/[0-9]/.test(cj)) break;
          j++;
        }
        const raw = src.slice(i, j);
        const value = Number(raw);
        if (Number.isNaN(value)) throw new Error(`ExpressionParser: invalid number '${raw}'`);
        pushToken({ type: 'number', value });
        i = j;
        continue;
      }
      if (isIdentStart(ch)) {
        let j = i + 1;
        while (j < src.length && isIdentChar(src[j])) j++;
        pushToken({ type: 'identifier', value: src.slice(i, j) });
        i = j;
        continue;
      }
      throw new Error(`ExpressionParser: unexpected character '${ch}'`);
    }
    return tokens;
  }

  static parseExpression(input, opts = {}) {
    const tokens = Array.isArray(input) ? input : ExpressionParser.tokenize(input, opts);
    let index = 0;
    const { maxDepth = 64 } = opts;

    const parseNode = (depth = 0) => {
      if (depth > maxDepth) {
        throw new Error(`ExpressionParser: max depth ${maxDepth} exceeded`);
      }
      const tok = tokens[index];
      if (!tok) throw new Error('ExpressionParser: unexpected end of input');
      if (tok.type === 'number') {
        index++;
        return { value: tok.value };
      }
      if (tok.type === 'string') {
        index++;
        return { literal: tok.value };
      }
      if (tok.type === 'identifier') {
        const name = tok.value;
        index++;
        const next = tokens[index];
        if (next && next.type === '(') {
          index++; // consume '('
          const args = [];
          while (index < tokens.length && tokens[index].type !== ')') {
            args.push(parseNode(depth + 1));
            if (tokens[index] && tokens[index].type === ',') index++;
          }
          if (!tokens[index] || tokens[index].type !== ')') {
            throw new Error('ExpressionParser: missing closing ")"');
          }
          index++; // consume ')'
          return ExpressionParser._buildCall(name, args);
        }
        return { literal: name };
      }
      throw new Error(`ExpressionParser: unexpected token '${tok.type}'`);
    };

    const node = parseNode();
    if (index < tokens.length) {
      throw new Error('ExpressionParser: unexpected trailing tokens');
    }
    return node;
  }

  static _buildCall(name, args) {
    const upper = name.toUpperCase();
    if (upper === 'FACT') {
      if (args.length < 2) throw new Error('FACT requires subject and predicate');
      const subject = ExpressionParser._literalValue(args[0]);
      const predicate = ExpressionParser._literalValue(args[1]);
      const infer = args[2] ? ExpressionParser._truthy(args[2]) : false;
      return { fact: { subject, predicate, infer } };
    }
    if (upper === 'REL' || upper === 'RELATION') {
      if (args.length < 2) throw new Error('REL requires a relation name and at least one argument');
      const relation = ExpressionParser._literalValue(args[0]);
      let relArgs = args.slice(1);
      let infer = false;
      const last = relArgs[relArgs.length - 1];
      if (last && last.literal !== undefined && last.literal.toLowerCase() === 'infer') {
        infer = true;
        relArgs = relArgs.slice(0, -1);
      }
      relArgs = relArgs.map(ExpressionParser._literalValue);
      return { relation: { name: relation, args: relArgs, infer } };
    }
    const { op, domain } = ExpressionParser._resolveOp(name);
    return { op, domain, inputs: args };
  }

  static _resolveOp(name) {
    const trimmed = name.trim();
    if (trimmed.includes('.')) {
      const parts = trimmed.split('.');
      const op = parts[parts.length - 1].toUpperCase();
      return { op, domain: trimmed };
    }
    return { op: trimmed.toUpperCase(), domain: undefined };
  }

  static _literalValue(node) {
    if (node.literal !== undefined) return node.literal;
    if (node.value !== undefined) return String(node.value);
    throw new Error('ExpressionParser: expected literal argument');
  }

  static _truthy(node) {
    if (node.value !== undefined) return Boolean(node.value);
    if (node.literal !== undefined) {
      const v = node.literal.toLowerCase();
      return v === 'true' || v === 'infer' || v === 'yes';
    }
    return false;
  }

  static toTokenStream(exprString, { factResolver, maxTokens, maxDepth } = {}) {
    const expr = ExpressionParser.parseExpression(exprString, { maxTokens, maxDepth });
    return ExpressionParser.expressionToTokens(expr, { factResolver });
  }

  static expressionToTokens(expression, { factResolver } = {}) {
    if (expression.fact) {
      if (!factResolver) {
        throw new Error('ExpressionParser: fact resolver required to tokenize fact nodes');
      }
      const resolved = factResolver(expression.fact);
      return ExpressionParser._valueToken(resolved);
    }
    if (expression.relation) {
      if (!factResolver) {
        throw new Error('ExpressionParser: relation resolver required to tokenize relation nodes');
      }
      const resolved = factResolver(expression.relation);
      return ExpressionParser._valueToken(resolved);
    }
    if (expression.value !== undefined) {
      return ExpressionParser._valueToken(expression.value);
    }
    if (!expression.op) {
      throw new Error('ExpressionParser: expected op or value node');
    }
    const opToken = TOKEN[expression.op];
    if (opToken === undefined) {
      throw new Error(`ExpressionParser: unknown operator '${expression.op}'`);
    }
    const opEntry = { token: opToken };
    if (expression.domain) {
      opEntry.domain = expression.domain;
    }
    const children = expression.inputs.flatMap(child =>
      ExpressionParser.expressionToTokens(child, { factResolver })
    );
    return [opEntry, ...children];
  }

  static _valueToken(value) {
    if (value === 0) return [{ token: TOKEN.V0 }];
    if (value === 1) return [{ token: TOKEN.V1 }];
    return [{ token: TOKEN.VALUE, value }];
  }

  static domainHintsForOp(opName) {
    return OP_DOMAIN_HINTS[opName.toUpperCase()] || [];
  }
}

module.exports = ExpressionParser;
