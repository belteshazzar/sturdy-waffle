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
  cleaned = cleaned.replace(/\s*\([^)]*\)\s*$/, '');
  cleaned = cleaned.replace(/['"]/g, '');
  cleaned = cleaned.replace(/&/g, 'and');
  cleaned = cleaned.replace(/\s+/g, '_');
  cleaned = cleaned.replace(/[^A-Za-z0-9_.-]/g, '');
  cleaned = cleaned.replace(/_+/g, '_');
  return cleaned;
}

function normalizeKey(raw) {
  if (!raw || typeof raw !== 'string') return '';
  let cleaned = raw.trim().toLowerCase();
  cleaned = cleaned.replace(/['"]/g, '');
  cleaned = cleaned.replace(/&/g, 'and');
  cleaned = cleaned.replace(/[^a-z0-9]+/g, ' ');
  cleaned = cleaned.trim().replace(/\s+/g, ' ');
  return cleaned;
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
