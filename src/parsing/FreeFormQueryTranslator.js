'use strict';

const { normalizeEntity } = require('../ingestion/EntityNormalization');

function cleanQuestion(text) {
  let cleaned = text.trim();
  while (cleaned.endsWith('?')) {
    cleaned = cleaned.slice(0, -1);
  }
  return cleaned.trim();
}

function translateLine(line, opts = {}) {
  const cleaned = cleanQuestion(line);
  if (!cleaned) return null;

  const lower = cleaned.toLowerCase();

  if (lower.startsWith('who is ')) {
    const subjectText = cleaned.slice('who is '.length).trim();
    const subject = normalizeEntity(subjectText);
    if (subject) {
      return { kind: 'attribute', subject, attribute: opts.identityAttribute || 'description' };
    }
  }

  if (lower.startsWith('what is ')) {
    const subjectText = cleaned.slice('what is '.length).trim();
    const subject = normalizeEntity(subjectText);
    if (subject) {
      return { kind: 'attribute', subject, attribute: opts.identityAttribute || 'description' };
    }
  }

  if (lower.startsWith('what type of ')) {
    const subjectText = cleaned.slice('what type of '.length).trim();
    const subject = normalizeEntity(subjectText);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'type' };
    }
  }

  if (lower.startsWith('when was ') && lower.endsWith(' born')) {
    const subjectText = cleaned.slice('when was '.length, cleaned.length - ' born'.length).trim();
    const subject = normalizeEntity(subjectText);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'birthDate' };
    }
  }

  if (lower.startsWith('when is ') && lower.endsWith(' born')) {
    const subjectText = cleaned.slice('when is '.length, cleaned.length - ' born'.length).trim();
    const subject = normalizeEntity(subjectText);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'birthDate' };
    }
  }

  if (lower.startsWith('where was ') && lower.endsWith(' born')) {
    const subjectText = cleaned.slice('where was '.length, cleaned.length - ' born'.length).trim();
    const subject = normalizeEntity(subjectText);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'birthPlace' };
    }
  }

  if (lower.startsWith('where is ') && lower.endsWith(' born')) {
    const subjectText = cleaned.slice('where is '.length, cleaned.length - ' born'.length).trim();
    const subject = normalizeEntity(subjectText);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'birthPlace' };
    }
  }

  if (lower.startsWith('where is ') && lower.endsWith(' located')) {
    const subjectText = cleaned.slice('where is '.length, cleaned.length - ' located'.length).trim();
    const subject = normalizeEntity(subjectText);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'location' };
    }
  }

  if (lower.startsWith('where was ') && lower.endsWith(' located')) {
    const subjectText = cleaned.slice('where was '.length, cleaned.length - ' located'.length).trim();
    const subject = normalizeEntity(subjectText);
    if (subject) {
      return { kind: 'attribute', subject, attribute: 'location' };
    }
  }

  if (lower.startsWith('where is ')) {
    const subjectText = cleaned.slice('where is '.length).trim();
    const subject = normalizeEntity(subjectText);
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
