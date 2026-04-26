'use strict';

const { normalizeEntity, normalizeAttribute, normalizeRelation } = require('./EntityNormalization');
const { segmentWikipediaText, stripWikitext } = require('./WikipediaText');

function stripParentheticals(value) {
  if (!value) return '';
  let result = '';
  let depth = 0;
  for (const ch of value) {
    if (ch === '(') {
      depth += 1;
      continue;
    }
    if (ch === ')' && depth > 0) {
      depth -= 1;
      continue;
    }
    if (depth === 0) result += ch;
  }
  return result;
}

function collapseWhitespace(value) {
  if (!value) return '';
  let result = '';
  let lastWasSpace = false;
  for (const ch of value) {
    const isWhitespace = ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r';
    if (isWhitespace) {
      if (!lastWasSpace && result.length) {
        result += ' ';
        lastWasSpace = true;
      }
      continue;
    }
    result += ch;
    lastWasSpace = false;
  }
  return result;
}

function trimPunctuation(value) {
  let cleaned = value.trim();
  while (cleaned && [',', '–', '-', ':', ';', '.'].includes(cleaned[0])) {
    cleaned = cleaned.slice(1).trim();
  }
  while (cleaned && [',', '–', '-', ':', ';', '.'].includes(cleaned[cleaned.length - 1])) {
    cleaned = cleaned.slice(0, -1).trim();
  }
  return cleaned;
}

function cleanPhrase(value) {
  if (!value) return '';
  let cleaned = stripParentheticals(value);
  cleaned = collapseWhitespace(cleaned);
  cleaned = trimPunctuation(cleaned);
  return cleaned.trim();
}

function extractInfoboxAttributes(wikitext, subject, meta) {
  if (!wikitext) return [];
  const statements = [];
  const lower = wikitext.toLowerCase();
  const start = lower.indexOf('{{infobox');
  if (start < 0) return statements;
  let end = lower.indexOf('\n}}', start);
  if (end < 0) end = lower.indexOf('}}', start);
  if (end < 0) return statements;
  const infobox = wikitext.slice(start, end + 2);
  const lines = infobox.split(/\r?\n/);
  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('|')) return;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx < 0) return;
    const rawKey = trimmed.slice(1, eqIdx).trim();
    const rawValue = trimmed.slice(eqIdx + 1).trim();
    const attribute = normalizeAttribute(rawKey);
    if (!attribute) return;
    const value = cleanPhrase(stripWikitext(rawValue));
    if (!value) return;
    statements.push({
      kind: 'attribute',
      subject,
      attribute,
      value,
      meta: { ...meta, confidence: 0.8 },
    });
  });
  return statements;
}

function extractFromSentence(sentence, subjectAliases, subject, meta) {
  const statements = [];
  const normalizedSentence = sentence.trim();
  if (!normalizedSentence) return statements;

  const alias = subjectAliases.find(candidate =>
    normalizedSentence.toLowerCase().startsWith(candidate.toLowerCase())
  );
  if (!alias) return statements;

  let remainder = normalizedSentence.slice(alias.length).trim();
  if (remainder.startsWith('(')) {
    const closeIdx = remainder.indexOf(')');
    if (closeIdx >= 0) {
      remainder = remainder.slice(closeIdx + 1).trim();
    }
  }
  remainder = trimPunctuation(remainder);

  const typeMatch = remainder.match(/^(is|was)\s+(an?|the)\s+([^.;]+)/i);
  if (typeMatch) {
    const typeValue = cleanPhrase(typeMatch[3]);
    if (typeValue) {
      statements.push({
        kind: 'attribute',
        subject,
        attribute: 'type',
        value: typeValue,
        meta: { ...meta, confidence: 0.6 },
      });
    }
  }

  const bornInMatch = remainder.match(/^(was|is)\s+born\s+in\s+([^.;]+)/i);
  if (bornInMatch) {
    const place = cleanPhrase(bornInMatch[2]);
    if (place) {
      statements.push({
        kind: 'attribute',
        subject,
        attribute: 'birthPlace',
        value: place,
        meta: { ...meta, confidence: 0.7 },
      });
    }
  }

  const bornOnMatch = remainder.match(/^(was|is)\s+born\s+on\s+([^.;]+)/i);
  if (bornOnMatch) {
    const dateValue = cleanPhrase(bornOnMatch[2]);
    if (dateValue) {
      statements.push({
        kind: 'attribute',
        subject,
        attribute: 'birthDate',
        value: dateValue,
        meta: { ...meta, confidence: 0.7 },
      });
    }
  }

  const bornYearMatch = remainder.match(/^(was|is)\s+born\s+([0-9][^.;]*)/i);
  if (bornYearMatch) {
    const dateValue = cleanPhrase(bornYearMatch[2]);
    if (dateValue) {
      statements.push({
        kind: 'attribute',
        subject,
        attribute: 'birthDate',
        value: dateValue,
        meta: { ...meta, confidence: 0.7 },
      });
    }
  }

  const diedOnMatch = remainder.match(/^(died)\s+on\s+([^.;]+)/i);
  if (diedOnMatch) {
    const dateValue = cleanPhrase(diedOnMatch[2]);
    if (dateValue) {
      statements.push({
        kind: 'attribute',
        subject,
        attribute: 'deathDate',
        value: dateValue,
        meta: { ...meta, confidence: 0.7 },
      });
    }
  }

  const diedInMatch = remainder.match(/^(died)\s+in\s+([^.;]+)/i);
  if (diedInMatch) {
    const place = cleanPhrase(diedInMatch[2]);
    if (place) {
      statements.push({
        kind: 'attribute',
        subject,
        attribute: 'deathPlace',
        value: place,
        meta: { ...meta, confidence: 0.7 },
      });
    }
  }

  const locatedInMatch = remainder.match(/^(is|was)\s+(located\s+in|in)\s+([^.;]+)/i);
  if (locatedInMatch) {
    const location = cleanPhrase(locatedInMatch[3]);
    if (location) {
      const locationEntity = normalizeEntity(location);
      if (locationEntity) {
        statements.push({
          kind: 'relation',
          name: normalizeRelation('locatedIn'),
          args: [subject, locationEntity],
          value: true,
          meta: { ...meta, confidence: 0.6 },
        });
      }
      statements.push({
        kind: 'attribute',
        subject,
        attribute: 'location',
        value: location,
        meta: { ...meta, confidence: 0.6 },
      });
    }
  }

  const capitalMatch = remainder.match(/^(is|was)\s+the\s+capital\s+of\s+([^.;]+)/i);
  if (capitalMatch) {
    const target = cleanPhrase(capitalMatch[2]);
    if (target) {
      const targetEntity = normalizeEntity(target);
      if (targetEntity) {
        statements.push({
          kind: 'relation',
          name: normalizeRelation('capitalOf'),
          args: [subject, targetEntity],
          value: true,
          meta: { ...meta, confidence: 0.6 },
        });
      }
    }
  }

  const partOfMatch = remainder.match(/^(is|was)\s+part\s+of\s+([^.;]+)/i);
  if (partOfMatch) {
    const target = cleanPhrase(partOfMatch[2]);
    if (target) {
      const targetEntity = normalizeEntity(target);
      if (targetEntity) {
        statements.push({
          kind: 'relation',
          name: normalizeRelation('partOf'),
          args: [subject, targetEntity],
          value: true,
          meta: { ...meta, confidence: 0.6 },
        });
      }
    }
  }

  return statements;
}

function buildWikipediaStatements(article, opts = {}) {
  if (!article) throw new Error('buildWikipediaStatements: article is required');
  const subject = normalizeEntity(article.title || opts.title);
  if (!subject) throw new Error('buildWikipediaStatements: could not normalize subject title');

  const aliases = [
    article.title,
    article.displayTitle,
    article.normalizedTitle,
  ].filter(Boolean);

  const sourceBase = opts.sourceBase || 'wikipedia';
  const statements = [];

  const aliasEntities = aliases
    .map(alias => normalizeEntity(alias))
    .filter(alias => alias && alias !== subject);

  aliasEntities.forEach(alias => {
    statements.push({
      kind: 'relation',
      name: normalizeRelation('aliasOf'),
      args: [subject, alias],
      value: true,
      meta: { source: sourceBase, confidence: 1 },
    });
  });

  if (article.description) {
    statements.push({
      kind: 'attribute',
      subject,
      attribute: 'description',
      value: cleanPhrase(article.description),
      meta: { source: sourceBase, confidence: 0.95 },
    });
  }

  if (article.extract) {
    const summary = cleanPhrase(article.extract.split(/\r?\n/).join(' '));
    if (summary) {
      statements.push({
        kind: 'attribute',
        subject,
        attribute: 'summary',
        value: summary,
        meta: { source: sourceBase, confidence: 0.9 },
      });
    }
  }

  if (article.contentUrls?.desktop?.page) {
    statements.push({
      kind: 'attribute',
      subject,
      attribute: 'url',
      value: article.contentUrls.desktop.page,
      meta: { source: sourceBase, confidence: 0.9 },
    });
  }

  const sections = segmentWikipediaText(article);
  sections.forEach(section => {
    section.sentences.forEach((sentence, idx) => {
      const meta = {
        source: `${sourceBase}:${article.title || subject}#${section.title}`,
        confidence: 0.6,
        sentence: idx + 1,
      };
      statements.push(...extractFromSentence(sentence, aliases, subject, meta));
    });
  });

  statements.push(...extractInfoboxAttributes(article.wikitext, subject, { source: sourceBase }));

  return {
    subject,
    aliases,
    sections,
    statements,
  };
}

module.exports = {
  buildWikipediaStatements,
  extractInfoboxAttributes,
  extractFromSentence,
};
