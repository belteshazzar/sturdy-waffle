'use strict';

const ATTRIBUTE_ALIASES = new Map([
  ['birth date', 'birthDate'],
  ['birth_date', 'birthDate'],
  ['date of birth', 'birthDate'],
  ['birthplace', 'birthPlace'],
  ['birth place', 'birthPlace'],
  ['place of birth', 'birthPlace'],
  ['death date', 'deathDate'],
  ['death_date', 'deathDate'],
  ['date of death', 'deathDate'],
  ['deathplace', 'deathPlace'],
  ['death place', 'deathPlace'],
  ['place of death', 'deathPlace'],
  ['location', 'location'],
  ['located in', 'location'],
  ['summary', 'summary'],
  ['description', 'description'],
  ['type', 'type'],
]);

function normalizeEntity(raw) {
  if (!raw || typeof raw !== 'string') return '';
  let cleaned = raw.trim();
  if (cleaned.endsWith(')')) {
    const openIdx = cleaned.lastIndexOf('(');
    if (openIdx >= 0) {
      cleaned = cleaned.slice(0, openIdx).trim();
    }
  }
  cleaned = cleaned.split('"').join('').split('\'').join('');
  cleaned = cleaned.split('&').join('and');
  let result = '';
  let lastWasUnderscore = false;
  for (const ch of cleaned) {
    const code = ch.charCodeAt(0);
    const isWhitespace = ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r';
    const isAllowed =
      (code >= 48 && code <= 57) ||
      (code >= 65 && code <= 90) ||
      (code >= 97 && code <= 122) ||
      ch === '_' ||
      ch === '.' ||
      ch === '-';
    if (isWhitespace) {
      if (!lastWasUnderscore && result.length) {
        result += '_';
        lastWasUnderscore = true;
      }
      continue;
    }
    if (isAllowed) {
      result += ch;
      lastWasUnderscore = ch === '_';
      continue;
    }
    lastWasUnderscore = false;
  }
  return result;
}

function normalizeKey(raw) {
  if (!raw || typeof raw !== 'string') return '';
  let cleaned = raw.trim().toLowerCase();
  cleaned = cleaned.split('"').join('').split('\'').join('');
  cleaned = cleaned.split('&').join('and');
  let result = '';
  let lastWasSpace = false;
  for (const ch of cleaned) {
    const code = ch.charCodeAt(0);
    const isAlnum = (code >= 48 && code <= 57) || (code >= 97 && code <= 122);
    if (isAlnum) {
      result += ch;
      lastWasSpace = false;
      continue;
    }
    if (!lastWasSpace && result.length) {
      result += ' ';
      lastWasSpace = true;
    }
  }
  return result.trim();
}

function toCamelCase(text) {
  if (!text) return '';
  const parts = text.split(' ').filter(Boolean);
  if (parts.length === 0) return '';
  return parts[0] + parts.slice(1).map(part => part[0].toUpperCase() + part.slice(1)).join('');
}

function normalizeAttribute(raw) {
  if (!raw || typeof raw !== 'string') return '';
  const token = raw.trim();
  if (/^[A-Za-z0-9_.-]+$/.test(token) && /[A-Z]/.test(token)) {
    return token;
  }
  const cleaned = normalizeKey(raw);
  if (!cleaned) return '';
  if (ATTRIBUTE_ALIASES.has(cleaned)) return ATTRIBUTE_ALIASES.get(cleaned);
  return toCamelCase(cleaned);
}

function normalizeRelation(raw) {
  return normalizeAttribute(raw);
}

module.exports = {
  normalizeEntity,
  normalizeAttribute,
  normalizeRelation,
  ATTRIBUTE_ALIASES,
};
