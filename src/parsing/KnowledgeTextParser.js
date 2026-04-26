'use strict';

/**
 * KnowledgeTextParser — parse controlled knowledge text into structured
 * facts, attributes, relations, and queries.
 *
 * Supported statement formats (one per line):
 *   fact: bird canFly
 *   fact: penguin not canFly
 *   fact: penguin canFly = false
 *   attribute: apple color = red
 *   attribute: mars radius = 3390; type=numeric
 *   relation: parentOf(alice,bob) = true
 *
 * Supported query formats:
 *   Is bird canFly?
 *   What is apple color?
 *   Is parentOf(alice,bob)?
 *   fact? bird canFly
 *   attribute? apple color
 *   relation? parentOf(alice,bob)
 *   expr: AND(1,0)
 *   solve: AND(1,0)
 *
 * Metadata segments (optional):
 *   ; confidence=0.8; source=manual
 */
class KnowledgeTextParser {
  static parse(text, { mode = 'both', defaultSource = 'text' } = {}) {
    const statements = [];
    const queries = [];
    if (!text || !text.trim()) return { statements, queries };

    const lines = text.split(/\r?\n/);
    lines.forEach((raw, idx) => {
      const line = raw.trim();
      if (!line || line.startsWith('#') || line.startsWith('//')) return;

      if ((mode === 'queries' || mode === 'both') && KnowledgeTextParser._looksLikeQuery(line)) {
        const query = KnowledgeTextParser.parseQueryLine(line);
        query.line = idx + 1;
        queries.push(query);
        return;
      }

      if (mode === 'queries') return;
      const statement = KnowledgeTextParser.parseStatementLine(line, { defaultSource });
      statement.line = idx + 1;
      statements.push(statement);
    });

    return { statements, queries };
  }

  static _looksLikeQuery(line) {
    const lower = line.toLowerCase();
    return line.endsWith('?') ||
      lower.startsWith('?') ||
      lower.startsWith('query:') ||
      lower.startsWith('fact?') ||
      lower.startsWith('attribute?') ||
      lower.startsWith('attr?') ||
      lower.startsWith('relation?') ||
      lower.startsWith('rel?') ||
      lower.startsWith('expr:') ||
      lower.startsWith('eval:') ||
      lower.startsWith('evaluate:') ||
      lower.startsWith('solve:') ||
      lower.startsWith('is ') ||
      lower.startsWith('does ') ||
      lower.startsWith('what is ') ||
      lower.startsWith('infer:');
  }

  static parseStatementLine(line, { defaultSource = 'text' } = {}) {
    const { statement, meta } = KnowledgeTextParser._splitMeta(line, { defaultSource });
    const match = statement.match(/^([A-Za-z]+)\s*:\s*(.+)$/);
    if (!match) {
      throw new Error(`KnowledgeTextParser: invalid statement '${line}'`);
    }
    const kind = match[1].toLowerCase();
    const rest = match[2].trim();

    if (kind === 'fact') {
      return KnowledgeTextParser._parseFact(rest, meta);
    }
    if (kind === 'attribute' || kind === 'attr') {
      return KnowledgeTextParser._parseAttribute(rest, meta);
    }
    if (kind === 'relation' || kind === 'rel') {
      return KnowledgeTextParser._parseRelation(rest, meta);
    }

    throw new Error(`KnowledgeTextParser: unknown statement type '${kind}'`);
  }

  static parseQueryLine(line) {
    let cleaned = line.trim();
    if (cleaned.startsWith('?')) cleaned = cleaned.slice(1).trim();
    if (cleaned.endsWith('?')) cleaned = cleaned.slice(0, -1).trim();

    const lower = cleaned.toLowerCase();
    if (lower.startsWith('query:')) cleaned = cleaned.slice(6).trim();

    if (lower.startsWith('expr:') || lower.startsWith('eval:') || lower.startsWith('evaluate:')) {
      const expr = cleaned.split(':').slice(1).join(':').trim();
      return { kind: 'expression', mode: 'evaluate', expression: expr };
    }
    if (lower.startsWith('solve:')) {
      const expr = cleaned.split(':').slice(1).join(':').trim();
      return { kind: 'expression', mode: 'solve', expression: expr };
    }

    if (lower.startsWith('infer:')) {
      const rest = cleaned.slice(6).trim();
      return KnowledgeTextParser._parseQueryRest(rest, { infer: true });
    }
    if (lower.startsWith('fact?')) {
      const rest = cleaned.slice(5).trim();
      return KnowledgeTextParser._parseQueryRest(rest, {});
    }
    if (lower.startsWith('attribute?') || lower.startsWith('attr?')) {
      const rest = cleaned.replace(/^attr(?:ibute)?\?/i, '').trim();
      return KnowledgeTextParser._parseAttributeQuery(rest, {});
    }
    if (lower.startsWith('relation?') || lower.startsWith('rel?')) {
      const rest = cleaned.replace(/^rel(?:ation)?\?/i, '').trim();
      return KnowledgeTextParser._parseRelationQuery(rest, {});
    }
    if (lower.startsWith('what is ')) {
      const rest = cleaned.slice(8).trim();
      return KnowledgeTextParser._parseAttributeQuery(rest, {});
    }
    if (lower.startsWith('is ') || lower.startsWith('does ')) {
      const rest = cleaned.replace(/^(is|does)\s+/i, '');
      return KnowledgeTextParser._parseQueryRest(rest, {});
    }

    return KnowledgeTextParser._parseQueryRest(cleaned, {});
  }

  static _parseQueryRest(rest, { infer = false } = {}) {
    const relationMatch = rest.match(/^([A-Za-z0-9_.-]+)\s*\((.*)\)$/);
    if (relationMatch) {
      const name = relationMatch[1];
      const args = KnowledgeTextParser._parseArgs(relationMatch[2]);
      return { kind: 'relation', name, args, infer };
    }
    const tokens = rest.split(/\s+/).filter(Boolean);
    const inferIdx = tokens.findIndex(t => t.toLowerCase() === 'infer');
    if (inferIdx >= 0) {
      tokens.splice(inferIdx, 1);
      infer = true;
    }
    if (tokens.length >= 2) {
      return { kind: 'fact', subject: tokens[0], predicate: tokens[1], infer };
    }
    throw new Error(`KnowledgeTextParser: invalid query '${rest}'`);
  }

  static _parseAttributeQuery(rest) {
    const tokens = rest.split(/\s+/).filter(Boolean);
    if (tokens.length < 2) {
      throw new Error(`KnowledgeTextParser: invalid attribute query '${rest}'`);
    }
    return { kind: 'attribute', subject: tokens[0], attribute: tokens[1] };
  }

  static _parseRelationQuery(rest) {
    const relationMatch = rest.match(/^([A-Za-z0-9_.-]+)\s*\((.*)\)$/);
    if (!relationMatch) {
      throw new Error(`KnowledgeTextParser: invalid relation query '${rest}'`);
    }
    return { kind: 'relation', name: relationMatch[1], args: KnowledgeTextParser._parseArgs(relationMatch[2]) };
  }

  static _parseFact(rest, meta) {
    let value = true;
    let subject = null;
    let predicate = null;
    let left = rest;
    let rhs = null;
    if (rest.includes('=')) {
      const parts = rest.split('=');
      left = parts[0].trim();
      rhs = parts.slice(1).join('=').trim();
    }
    if (rhs !== null) {
      value = KnowledgeTextParser._parseBoolean(rhs);
    }
    const tokens = left.split(/\s+/).filter(Boolean);
    if (tokens.length < 2) {
      throw new Error(`KnowledgeTextParser: invalid fact '${rest}'`);
    }
    subject = tokens[0];
    if (tokens[1].toLowerCase() === 'not' || tokens[1] === '!') {
      value = false;
      predicate = tokens[2];
    } else {
      predicate = tokens[1];
    }
    if (predicate && predicate.startsWith('!')) {
      predicate = predicate.slice(1);
      value = false;
    }
    if (!predicate) {
      throw new Error(`KnowledgeTextParser: invalid fact '${rest}'`);
    }
    return { kind: 'fact', subject, predicate, value, meta };
  }

  static _parseAttribute(rest, meta) {
    if (!rest.includes('=')) {
      throw new Error(`KnowledgeTextParser: attribute requires '=' in '${rest}'`);
    }
    const parts = rest.split('=');
    const left = parts[0].trim();
    const rhs = parts.slice(1).join('=').trim();
    const tokens = left.split(/\s+/).filter(Boolean);
    if (tokens.length < 2) {
      throw new Error(`KnowledgeTextParser: invalid attribute '${rest}'`);
    }
    const subject = tokens[0];
    const attribute = tokens[1];
    const value = KnowledgeTextParser._parseValue(rhs, meta);
    return { kind: 'attribute', subject, attribute, value, meta };
  }

  static _parseRelation(rest, meta) {
    let value = true;
    let left = rest;
    let rhs = null;
    if (rest.includes('=')) {
      const parts = rest.split('=');
      left = parts[0].trim();
      rhs = parts.slice(1).join('=').trim();
    }
    if (rhs !== null) {
      value = KnowledgeTextParser._parseBoolean(rhs);
    }
    if (left.toLowerCase().startsWith('not ')) {
      left = left.slice(4).trim();
      value = false;
    }
    if (left.startsWith('!')) {
      left = left.slice(1).trim();
      value = false;
    }
    const match = left.match(/^([A-Za-z0-9_.-]+)\s*\((.*)\)$/);
    if (!match) {
      throw new Error(`KnowledgeTextParser: invalid relation '${rest}'`);
    }
    const name = match[1];
    const args = KnowledgeTextParser._parseArgs(match[2]);
    return { kind: 'relation', name, args, value, meta };
  }

  static _parseArgs(argText) {
    return argText.split(',').map(arg => arg.trim()).filter(Boolean);
  }

  static _splitMeta(line, { defaultSource }) {
    const segments = line.split(';').map(seg => seg.trim()).filter(Boolean);
    const statement = segments[0];
    const meta = {};
    segments.slice(1).forEach(seg => {
      const [rawKey, ...rest] = seg.split('=');
      if (!rawKey || rest.length === 0) return;
      const key = rawKey.trim().toLowerCase();
      const value = rest.join('=').trim();
      if (!value) return;
      if (key === 'confidence') {
        const num = Number(value);
        if (!Number.isNaN(num)) meta.confidence = num;
      } else if (key === 'source') {
        meta.source = KnowledgeTextParser._stripQuotes(value);
      } else if (key === 'type') {
        meta.type = value.toLowerCase();
      }
    });
    if (!meta.source && defaultSource) meta.source = defaultSource;
    return { statement, meta };
  }

  static _parseBoolean(value) {
    const norm = KnowledgeTextParser._stripQuotes(value).toLowerCase();
    if (['true', 't', 'yes', 'y', '1'].includes(norm)) return true;
    if (['false', 'f', 'no', 'n', '0'].includes(norm)) return false;
    throw new Error(`KnowledgeTextParser: invalid boolean '${value}'`);
  }

  static _parseValue(value, meta) {
    const stripped = KnowledgeTextParser._stripQuotes(value);
    if (meta && meta.type === 'numeric') {
      const num = Number(stripped);
      if (Number.isNaN(num)) {
        throw new Error(`KnowledgeTextParser: invalid numeric value '${value}'`);
      }
      return num;
    }
    const num = Number(stripped);
    if (!Number.isNaN(num) && stripped !== '') {
      return num;
    }
    return stripped;
  }

  static _stripQuotes(value) {
    const trimmed = value.trim();
    if ((trimmed.startsWith('"') && trimmed.endsWith('"')) ||
        (trimmed.startsWith('\'') && trimmed.endsWith('\''))) {
      return trimmed.slice(1, -1);
    }
    return trimmed;
  }
}

module.exports = KnowledgeTextParser;
