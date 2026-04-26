'use strict';

const NeuralNetwork = require('../brain/NeuralNetwork');

/**
 * StringEncoder — a character-level neural model that maps human-readable
 * expression strings to sequences of integer token IDs (and optionally to
 * embedding vectors), bypassing the need for callers to know the integer TOKEN
 * vocabulary.
 *
 * Neuroscience analogue: the fusiform / visual word-form area that learns to
 * recognise written words and map them to conceptual representations —
 * character patterns become symbols, symbols become actions.
 *
 * Expression string format
 * ────────────────────────
 *  Tokens are separated by any mix of spaces, commas, and parentheses.
 *  Examples (prefix notation):
 *    "AND(OR(1,0), NOT(0))"
 *    "AND OR 1 0 NOT 0"         (space-separated)
 *    "XOR(NOT(1), NAND(0,1))"
 *
 *  Both formats tokenise to the same word sequence, e.g.
 *    ["AND", "OR", "1", "0", "NOT", "0"]
 *
 * Character vocabulary
 * ────────────────────
 *  Upper-case A-Z, digits 0-9, and a padding character.
 *  Each word is padded / truncated to `maxWordLen` characters, then one-hot
 *  encoded to form a feature vector of size `charVocabSize * maxWordLen`.
 *
 * Learned model
 * ─────────────
 *  A small feedforward network maps the character features of a single word to
 *  a probability distribution over the TOKEN vocabulary (V0…XNOR).
 *  Training uses the word→tokenId pairs from the decomposition curriculum.
 *
 * Static helper
 * ─────────────
 *  StringEncoder.toTokenIds(exprString) provides a purely deterministic
 *  (non-neural) conversion for use as a fallback and in testing.
 */
class StringEncoder {
  // Character vocabulary: A-Z, 0-9, plus one catch-all padding character.
  static CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_';
  static CHAR_VOCAB_SIZE = StringEncoder.CHARS.length; // 37

  /**
   * @param {object} opts
   * @param {number} [opts.maxWordLen=4]      Chars per word after pad/truncate
   * @param {number} [opts.hiddenSize=32]     Hidden neurons in the word encoder
    * @param {number} [opts.vocabSize=16]       Output classes (TOKEN vocabulary)
   * @param {number} [opts.learningRate=0.05]
   */
  constructor({
    maxWordLen  = 4,
    hiddenSize  = 32,
    vocabSize   = 16,
    learningRate = 0.05,
  } = {}) {
    this.maxWordLen   = maxWordLen;
    this.vocabSize    = vocabSize;

    const inputSize = StringEncoder.CHAR_VOCAB_SIZE * maxWordLen;

    this.network = new NeuralNetwork({
      architecture:     [inputSize, hiddenSize, vocabSize],
      learningRate,
      hiddenActivation: 'tanh',
      outputActivation: 'sigmoid',
    });
  }

  // ── Tokenisation helpers ──────────────────────────────────────────────────

  /**
   * Split an expression string into an ordered array of word tokens.
   * Parentheses, commas, and whitespace are all treated as separators.
   *
   * @param {string} exprString
   * @returns {string[]}
   */
  static splitWords(exprString) {
    return exprString.trim().split(/[\s,()]+/).filter(Boolean);
  }

  /**
   * Deterministic (non-neural) conversion of an expression string to integer
   * token IDs using the canonical TOKEN vocabulary.
   *
   * Throws on any unrecognised word.
   *
   * @param {string} exprString
   * @returns {number[]}
   */
  static toTokenIds(exprString) {
    const { TOKEN } = require('./tokens');
    return StringEncoder.splitWords(exprString).map(word => {
      const upper = word.toUpperCase();
      if (Object.prototype.hasOwnProperty.call(TOKEN, upper)) {
        return TOKEN[upper];
      }
      const num = Number(word);
      if (!Number.isNaN(num)) {
        if (num === 0) return TOKEN.V0;
        if (num === 1) return TOKEN.V1;
        return { token: TOKEN.VALUE, value: num };
      }
      throw new Error(`StringEncoder: unknown token '${word}'`);
    });
  }

  // ── Feature encoding ──────────────────────────────────────────────────────

  /**
   * Convert a single word to a fixed-length character one-hot feature vector.
   * The word is upper-cased, padded to `maxWordLen` with '_', then truncated.
   *
   * @param {string} word
   * @returns {number[]}  Length = CHAR_VOCAB_SIZE * maxWordLen
   */
  wordToFeature(word) {
    const chars  = StringEncoder.CHARS;
    const padded = word.toUpperCase().padEnd(this.maxWordLen, '_').slice(0, this.maxWordLen);
    const vec    = new Array(StringEncoder.CHAR_VOCAB_SIZE * this.maxWordLen).fill(0);
    for (let i = 0; i < this.maxWordLen; i++) {
      const idx = chars.indexOf(padded[i]);
      if (idx >= 0) {
        vec[i * StringEncoder.CHAR_VOCAB_SIZE + idx] = 1;
      }
    }
    return vec;
  }

  // ── Neural inference ──────────────────────────────────────────────────────

  /**
   * Predict the token ID for a single word using the learned model.
   * Returns the argmax of the output distribution.
   *
   * @param {string} word
   * @returns {number}  Predicted token ID
   */
  predictTokenId(word) {
    const scores  = this.network.predict(this.wordToFeature(word));
    let   maxIdx  = 0;
    let   maxVal  = -Infinity;
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > maxVal) { maxVal = scores[i]; maxIdx = i; }
    }
    return maxIdx;
  }

  /**
   * Encode a full expression string to an array of predicted token IDs.
   *
   * @param {string} exprString
   * @returns {number[]}
   */
  encode(exprString) {
    const { TOKEN } = require('./tokens');
    return StringEncoder.splitWords(exprString).map(word => {
      const num = Number(word);
      if (!Number.isNaN(num)) {
        if (num === 0) return TOKEN.V0;
        if (num === 1) return TOKEN.V1;
        return { token: TOKEN.VALUE, value: num };
      }
      return this.predictTokenId(word);
    });
  }

  // ── Training ──────────────────────────────────────────────────────────────

  /**
   * Train the encoder to map word strings to token IDs.
   *
   * @param {Array<{ word: string, tokenId: number }>} examples
   * @param {number} [epochs=40]
   */
  train(examples, epochs = 40) {
    if (examples.length === 0) return;
    const samples = examples.map(({ word, tokenId }) => {
      const id = tokenId && typeof tokenId === 'object' ? tokenId.token : tokenId;
      const target = new Array(this.vocabSize).fill(0);
      if (id >= 0 && id < this.vocabSize) target[id] = 1;
      return { input: this.wordToFeature(word), output: target };
    });
    this.network.train(samples, epochs);
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      maxWordLen:  this.maxWordLen,
      vocabSize:   this.vocabSize,
      network:     this.network.toJSON(),
    };
  }

  static fromJSON(data) {
    const enc = new StringEncoder({
      maxWordLen:  data.maxWordLen,
      vocabSize:   data.vocabSize,
    });
    enc.network = NeuralNetwork.fromJSON(data.network);
    return enc;
  }
}

module.exports = StringEncoder;
