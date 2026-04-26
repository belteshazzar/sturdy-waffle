'use strict';

const { normalizeEntity } = require('../ingestion/EntityNormalization');

function cleanQuestion(text) {
  return text.replace(/[?]+$/, '').trim();
}

function translateLine(line, opts = {}) {
  const cleaned = cleanQuestion(line);
  if (!cleaned) return null;

  const lower = cleaned.toLowerCase();

  let match = lower.match(/^(who|what)\s+is\s+(.+)$/i);
  if (match) {
    const subject = normalizeEntity(match[2]);
    if (subject) {
      return { kind: 'attribute', subject, attribute: opts.identityAttribute || 'description' };
    }
  }

  match = lower.match(/^what\s+type\s+of\s+(.+)$/i);
  if (match) {
    const subject = normalizeEntity(match[1]);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'type' };
    }
  }

  match = lower.match(/^(when\s+was|when\s+is)\s+(.+)\s+born$/i);
  if (match) {
    const subject = normalizeEntity(match[2]);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'birthDate' };
    }
  }

  match = lower.match(/^(where\s+was|where\s+is)\s+(.+)\s+born$/i);
  if (match) {
    const subject = normalizeEntity(match[2]);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'birthPlace' };
    }
  }

  match = lower.match(/^(where\s+is|where\s+was)\s+(.+)\s+located$/i);
  if (match) {
    const subject = normalizeEntity(match[2]);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'location' };
    }
  }

  match = lower.match(/^where\s+is\s+(.+)$/i);
  if (match) {
    const subject = normalizeEntity(match[1]);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'location' };
    }
  }

  return null;
}

function translate(text, opts = {}) {
  if (!text) return [];
  const skipLines = new Set(opts.skipLines || []);
  const lines = text.split(/\r?\n/);
  const queries = [];
  lines.forEach((raw, idx) => {
    const lineNumber = idx + 1;
    if (skipLines.has(lineNumber)) return;
    const trimmed = raw.trim();
    if (!trimmed) return;
    const query = translateLine(trimmed, opts);
    if (query) {
      query.line = lineNumber;
      queries.push(query);
    }
  });
  return queries;
}

module.exports = {
  translate,
  translateLine,
};
