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
   * @param {object}   [opts.networkConfig]  Override the default network activations.
   *   e.g. `{ hiddenActivation: 'tanh', outputActivation: 'linear' }` for regression.
   * @param {'classification'|'regression'} [opts.mode='classification']
   *   'classification' uses binary-threshold accuracy; 'regression' uses
   *   tolerance-based continuous accuracy.
   * @param {object}   [opts.normalise]   Optional scaling applied to raw data before
   *   feeding the network.  Both fields are independent and optional:
   *   `{ inputRange: [min, max], outputRange: [min, max] }`
   *   Values are scaled from the given range to [0, 1] for training and the
   *   inverse is applied to predictions so callers always work in raw units.
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
    networkConfig = null,
    mode = 'classification',
    normalise = null,
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
    this.networkConfig  = networkConfig;
    this.mode           = mode;
    this.normalise      = normalise;
  }
}

module.exports = Lesson;
