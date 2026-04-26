'use strict';

const { fetchWikipediaArticle } = require('./WikipediaClient');
const { buildWikipediaStatements } = require('./WikipediaExtractor');
const { segmentWikipediaText } = require('./WikipediaText');
const {
  ingestWikipediaArticle,
  buildKnowledgeLine,
  saveFactBase,
  loadFactBase,
  evaluateQuestionSet,
} = require('./WikipediaIngestion');
const {
  normalizeEntity,
  normalizeAttribute,
  normalizeRelation,
  ATTRIBUTE_ALIASES,
} = require('./EntityNormalization');

module.exports = {
  fetchWikipediaArticle,
  buildWikipediaStatements,
  segmentWikipediaText,
  ingestWikipediaArticle,
  buildKnowledgeLine,
  saveFactBase,
  loadFactBase,
  evaluateQuestionSet,
  normalizeEntity,
  normalizeAttribute,
  normalizeRelation,
  ATTRIBUTE_ALIASES,
};
