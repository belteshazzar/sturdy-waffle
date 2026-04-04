'use strict';

/**
 * A Lesson is the unit of knowledge presented to the Brain.
 *
 * Each lesson belongs to exactly one domain (e.g. "boolean.AND") and carries
 * its own training data.  The Brain spawns a new BrainRegion for any domain it
 * hasn't encountered before.
 */
class Lesson {
  /**
   * @param {object}   opts
   * @param {string}   opts.name          Human-readable name
   * @param {string}   opts.domain        Dot-notation domain key, e.g. "boolean.AND"
   * @param {string}   [opts.description] Optional description
   * @param {Array<{input: number[], output: number[]}>} opts.trainingData
   * @param {Array<{input: number[], output: number[]}>} [opts.validationData]
   *   Defaults to trainingData when omitted
   * @param {number}   [opts.inputSize]   Inferred from first training sample
   * @param {number}   [opts.outputSize]  Inferred from first training sample
   * @param {string[]} [opts.tags]        Arbitrary labels for grouping
   */
  constructor({
    name,
    domain,
    description = '',
    trainingData,
    validationData,
    inputSize,
    outputSize,
    tags = [],
  }) {
    if (!name)                              throw new Error('Lesson must have a name');
    if (!domain)                            throw new Error('Lesson must have a domain');
    if (!Array.isArray(trainingData) || trainingData.length === 0) {
      throw new Error('Lesson must have a non-empty trainingData array');
    }

    this.name           = name;
    this.domain         = domain;
    this.description    = description;
    this.trainingData   = trainingData;
    this.validationData = validationData || trainingData;
    this.inputSize      = inputSize  ?? trainingData[0].input.length;
    this.outputSize     = outputSize ?? trainingData[0].output.length;
    this.tags           = tags;
  }
}

module.exports = Lesson;
