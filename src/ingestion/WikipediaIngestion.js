'use strict';

const fs = require('fs');
const path = require('path');
const { fetchWikipediaArticle } = require('./WikipediaClient');
const { buildWikipediaStatements } = require('./WikipediaExtractor');
const { normalizeEntity, normalizeAttribute, normalizeRelation } = require('./EntityNormalization');

function quoteValue(value) {
  if (typeof value === 'number') return value.toString();
  const str = String(value);
  if (/^[A-Za-z0-9_.-]+$/.test(str)) return str;
  return `"${str.replace(/"/g, '\\"')}"`;
}

function buildSource(meta, fallback) {
  if (!meta) return fallback;
  if (meta.source) return meta.source;
  return fallback;
}

function buildKnowledgeLine(statement, limits) {
  const maxLineLength = limits?.maxLineLength || 1000;
  let base = '';
  let metaSuffix = '';

  const meta = statement.meta || {};
  const source = buildSource(meta, 'wikipedia');
  if (source) metaSuffix += `; source="${source.replace(/"/g, '\\"')}"`;
  if (meta.confidence != null) metaSuffix += `; confidence=${meta.confidence}`;
  if (meta.type) metaSuffix += `; type=${meta.type}`;

  if (statement.kind === 'fact') {
    const subject = normalizeEntity(statement.subject);
    const predicate = normalizeAttribute(statement.predicate);
    base = `fact: ${subject} ${predicate}`;
    if (statement.value === false || statement.value === 0) {
      base += ' = false';
    }
  } else if (statement.kind === 'attribute') {
    const subject = normalizeEntity(statement.subject);
    const attribute = normalizeAttribute(statement.attribute);
    base = `attribute: ${subject} ${attribute} = ${quoteValue(statement.value)}`;
  } else if (statement.kind === 'relation') {
    const name = normalizeRelation(statement.name);
    const args = (statement.args || []).map(arg => normalizeEntity(arg));
    base = `relation: ${name}(${args.join(',')}) = ${statement.value ? 'true' : 'false'}`;
  }

  let line = `${base}${metaSuffix}`;
  if (line.length <= maxLineLength) return line;

  if (statement.kind === 'attribute' && typeof statement.value === 'string') {
    const baseWithoutValue = `attribute: ${normalizeEntity(statement.subject)} ${normalizeAttribute(statement.attribute)} = `;
    const suffixLength = metaSuffix.length;
    const available = maxLineLength - baseWithoutValue.length - suffixLength - 2;
    if (available > 4) {
      const truncated = statement.value.slice(0, available - 1).trim() + '…';
      line = `${baseWithoutValue}${quoteValue(truncated)}${metaSuffix}`;
      if (line.length <= maxLineLength) return line;
    }
  }
  return null;
}

function chunkLines(lines, maxLines) {
  if (!maxLines || maxLines <= 0) return [lines];
  const chunks = [];
  for (let i = 0; i < lines.length; i += maxLines) {
    chunks.push(lines.slice(i, i + maxLines));
  }
  return chunks;
}

async function ingestWikipediaArticle(brain, title, opts = {}) {
  if (!brain) throw new Error('ingestWikipediaArticle: brain is required');
  if (!title) throw new Error('ingestWikipediaArticle: title is required');

  const {
    article: providedArticle,
    retrain = true,
    language = 'en',
    includeHtml = false,
    includeWikitext = false,
    name,
  } = opts;

  const article = providedArticle || await fetchWikipediaArticle(title, {
    language,
    includeHtml,
    includeWikitext,
    userAgent: opts.userAgent,
  });

  const sourceBase = `wikipedia:${article.title || title}`;
  const extraction = buildWikipediaStatements(article, { sourceBase });
  const limits = brain.config?.inputLimits || {};

  const lines = extraction.statements
    .map(statement => buildKnowledgeLine(statement, limits))
    .filter(Boolean);

  const chunks = chunkLines(lines, limits.maxLines || lines.length);
  const results = [];
  const trainedDomains = new Set();

  chunks.forEach(chunk => {
    if (chunk.length === 0) return;
    const text = chunk.join('\n');
    const result = brain.learnText(text, {
      name: name || `Wikipedia:${article.title || title}`,
      source: sourceBase,
      retrain,
    });
    results.push(result);
    result.trainedDomains.forEach(domain => trainedDomains.add(domain));
  });

  return {
    article,
    subject: extraction.subject,
    lines,
    statements: extraction.statements,
    results,
    trainedDomains: [...trainedDomains],
  };
}

function saveFactBase(factBase, filepath) {
  if (!factBase) throw new Error('saveFactBase: factBase is required');
  const dir = path.dirname(filepath);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(filepath, JSON.stringify(factBase.toJSON(), null, 2), 'utf8');
}

function loadFactBase(filepath, FactBase) {
  if (!FactBase) throw new Error('loadFactBase: FactBase constructor is required');
  const raw = JSON.parse(fs.readFileSync(filepath, 'utf8'));
  return FactBase.fromJSON(raw);
}

function evaluateQuestionSet(brain, questions, opts = {}) {
  if (!Array.isArray(questions)) return { accuracy: 0, results: [] };
  let correct = 0;
  const results = questions.map(question => {
    const answers = brain.answerFreeForm(question.question, opts);
    const value = answers[0]?.value;
    const isCorrect = value === question.expected;
    if (isCorrect) correct += 1;
    return {
      question: question.question,
      expected: question.expected,
      value,
      correct: isCorrect,
    };
  });
  const accuracy = results.length ? correct / results.length : 0;
  return { accuracy, results };
}

module.exports = {
  ingestWikipediaArticle,
  buildKnowledgeLine,
  saveFactBase,
  loadFactBase,
  evaluateQuestionSet,
};
