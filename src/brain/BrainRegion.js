'use strict';

const { EventEmitter } = require('events');
const NeuralNetwork    = require('./NeuralNetwork');

/**
 * A BrainRegion encapsulates a single neural network together with metadata
 * that governs its lifecycle:
 *
 *  • domain       – dot-notation identifier, e.g. "boolean.AND"
 *  • plasticity   – 1.0 = fully malleable (new), 0.1 = consolidated (learned)
 *  • accuracy     – fraction of validation samples answered correctly
 *  • trained      – true once the region meets its target accuracy
 *
 * The region drives its own training loop, emitting events at each milestone so
 * that the parent Brain (or a visualisation layer) can react.
 *
 * Plasticity affects the effective learning rate:
 *   effectiveLR = baseLearningRate × plasticity
 * A freshly spawned region starts with high plasticity and a high effective LR.
 * Once consolidated (trained) the region's plasticity drops, making it stable.
 *
 * When the network's accuracy plateaus, the region mutates its architecture
 * (adds neurons / adds layers / resets weights) and continues training.
 */
class BrainRegion extends EventEmitter {
  /**
   * @param {object} opts
   * @param {string}  opts.domain   Dot-notation domain key
   * @param {object}  opts.lesson   Lesson object (trainingData, validationData, inputSize, outputSize)
   * @param {object}  [opts.config] Override default training hyper-parameters
   */
  constructor({ domain, lesson, config = {} }) {
    super();

    this.domain          = domain;
    this.trained         = false;
    this.plasticity      = 1.0;
    this.accuracy        = 0;
    this.trainingHistory = [];
    this.mutationCount   = 0;

    this.config = {
      targetAccuracy:           0.99,
      maxMutations:             12,
      epochsPerRound:           500,
      maxEpochsTotal:           30000,
      accuracyPlateauThreshold: 0.005,
      mutationPatience:         3,      // rounds without improvement before mutating
      baseLearningRate:         0.2,
      regressionTolerance:      0.05,   // tolerance for regression accuracy checks
      ...config,
    };

    this.lesson = lesson;
    this._initializeNetwork(lesson);
  }

  // ── Network initialisation ────────────────────────────────────────────────

  /**
   * Size the initial hidden layer relative to the problem's input complexity.
   * Reads optional lesson.networkConfig to select activations, and pre-computes
   * normalised training / validation data if lesson.normalise is provided.
   */
  _initializeNetwork(lesson) {
    const inSize  = lesson.inputSize;
    const outSize = lesson.outputSize;
    const COMPLEXITY_SCALE_FACTOR = 2;
    const sampleCount = Array.isArray(lesson.trainingData) ? lesson.trainingData.length : 0;
    const isRegression = lesson.mode === 'regression';
    let complexityBoost = 0;
    if (isRegression) {
      // Sub-linear complexity boost: larger datasets justify more starting
      // capacity, but sqrt growth avoids over-scaling hidden size.
      complexityBoost = Math.ceil(
        Math.sqrt(Math.max(1, sampleCount)) / COMPLEXITY_SCALE_FACTOR
      );
    }
    // Start with higher capacity for regression to better capture continuous math functions.
    const hidden = isRegression
      ? Math.max(8, Math.min(64, inSize * 8 + complexityBoost))
      : Math.max(6, Math.min(32, inSize * 6));

    const netConfig = lesson.networkConfig || {};
    this.network = new NeuralNetwork({
      architecture:     [inSize, hidden, outSize],
      learningRate:     this.config.baseLearningRate,
      hiddenActivation: netConfig.hiddenActivation || 'sigmoid',
      outputActivation: netConfig.outputActivation || 'sigmoid',
    });

    // Pre-compute normalised copies of the training and validation sets.
    // When lesson.normalise is null these are identical to the raw sets.
    this._normTrainingData   = lesson.trainingData.map(s => this._normalizeSample(s));
    const rawVal             = lesson.validationData || lesson.trainingData;
    this._normValidationData = rawVal.map(s => this._normalizeSample(s));
  }

  // ── Normalisation helpers ─────────────────────────────────────────────────

  /** Scale a single value from [min, max] → [0, 1]. */
  _scaleValue(v, range) {
    const span = range[1] - range[0];
    if (span === 0) return 0;
    return (v - range[0]) / span;
  }

  /** Invert the scaling: [0, 1] → [min, max]. */
  _unscaleValue(v, range) {
    return v * (range[1] - range[0]) + range[0];
  }

  _normalizeInput(input) {
    const range = this.lesson.normalise && this.lesson.normalise.inputRange;
    if (!range) return input;
    return input.map(v => this._scaleValue(v, range));
  }

  _normalizeOutput(output) {
    const range = this.lesson.normalise && this.lesson.normalise.outputRange;
    if (!range) return output;
    return output.map(v => this._scaleValue(v, range));
  }

  _denormalizeOutput(output) {
    const range = this.lesson.normalise && this.lesson.normalise.outputRange;
    if (!range) return output;
    return output.map(v => this._unscaleValue(v, range));
  }

  _normalizeSample(sample) {
    return {
      input:  this._normalizeInput(sample.input),
      output: this._normalizeOutput(sample.output),
    };
  }

  // ── Accuracy selection ────────────────────────────────────────────────────

  /**
   * Choose the appropriate accuracy metric based on the lesson mode.
   * 'regression' uses a tolerance-based continuous metric;
   * 'classification' (default) uses exact binary matching.
   */
  _measureAccuracy(data) {
    if (this.lesson.mode === 'regression') {
      return this.network.regressionAccuracy(data, this.config.regressionTolerance);
    }
    return this.network.accuracy(data);
  }

  // ── Training loop ─────────────────────────────────────────────────────────

  /**
   * Train the region until the target accuracy is reached or all budget is
   * exhausted.  Mutations are applied automatically when progress stalls.
   *
   * @returns {{ trained: boolean, accuracy: number, mutationCount: number, totalEpochs: number }}
   */
  train() {
    // Fast-path: already trained and accuracy still holds
    if (this.trained) {
      this.accuracy = this._measureAccuracy(this._normValidationData);
      if (this.accuracy >= this.config.targetAccuracy) {
        return { trained: true, accuracy: this.accuracy, mutationCount: this.mutationCount, totalEpochs: 0 };
      }
      // Accuracy degraded (e.g. after deserialization rounding) – retrain
      this.trained = false;
    }

    let bestAccuracy            = 0;
    let roundsWithoutImprovement = 0;
    let totalEpochs             = 0;

    this.emit('training:start', {
      domain:       this.domain,
      architecture: [...this.network.architecture],
    });

    while (totalEpochs < this.config.maxEpochsTotal) {
      // Reduce effective LR as plasticity decreases (during retraining)
      const effectiveLR = Math.max(0.001, this.config.baseLearningRate * this.plasticity);
      this.network.learningRate = effectiveLR;

      const { finalLoss } = this.network.train(this._normTrainingData, this.config.epochsPerRound);
      totalEpochs += this.config.epochsPerRound;

      this.accuracy = this._measureAccuracy(this._normValidationData);

      const snapshot = {
        epoch:        totalEpochs,
        loss:         finalLoss,
        accuracy:     this.accuracy,
        architecture: [...this.network.architecture],
        plasticity:   this.plasticity,
      };
      this.trainingHistory.push(snapshot);

      this.emit('training:progress', {
        domain:   this.domain,
        ...snapshot,
      });

      if (this.accuracy >= this.config.targetAccuracy) {
        this.trained = true;
        break;
      }

      if (this.accuracy > bestAccuracy + this.config.accuracyPlateauThreshold) {
        bestAccuracy             = this.accuracy;
        roundsWithoutImprovement = 0;
      } else {
        roundsWithoutImprovement++;
      }

      if (
        roundsWithoutImprovement >= this.config.mutationPatience &&
        this.mutationCount < this.config.maxMutations
      ) {
        this._mutate();
        roundsWithoutImprovement = 0;
        bestAccuracy             = 0; // reset so we detect new progress post-mutation
      }
    }

    if (this.trained) {
      this._consolidate();
    }

    const result = {
      trained:       this.trained,
      accuracy:      this.accuracy,
      mutationCount: this.mutationCount,
      totalEpochs,
    };

    this.emit('training:complete', {
      domain: this.domain,
      architecture: [...this.network.architecture],
      ...result,
    });

    return result;
  }

  // ── Mutation ──────────────────────────────────────────────────────────────

  _mutate() {
    this.mutationCount++;
    const roll     = Math.random();
    const numHidden = this.network.layers.length - 1;
    let mutationType;

    if (roll < 0.35 && numHidden > 0) {
      // Add neurons to a random hidden layer
      const layerIdx = Math.floor(Math.random() * numHidden);
      const count    = Math.floor(Math.random() * 4) + 2;
      this.network.addNeuronsToLayer(layerIdx, count);
      mutationType = `addNeurons(layer=${layerIdx}, count=${count})`;
    } else if (roll < 0.60) {
      // Add a new hidden layer
      const size = Math.floor(Math.random() * 8) + 4;
      this.network.addHiddenLayer(size);
      mutationType = `addLayer(size=${size})`;
    } else if (roll < 0.80) {
      // Boost learning rate temporarily
      this.config.baseLearningRate = Math.min(0.5, this.config.baseLearningRate * 1.5);
      mutationType = `boostLR(lr=${this.config.baseLearningRate.toFixed(4)})`;
    } else {
      // Hard reset: re-initialise all weights (keep architecture)
      const arch = [...this.network.architecture];
      this.network = new NeuralNetwork({
        architecture:     arch,
        learningRate:     this.config.baseLearningRate,
        hiddenActivation: this.network.hiddenActivation,
        outputActivation: this.network.outputActivation,
      });
      mutationType = `reinitialise(arch=[${arch}])`;
    }

    this.emit('mutation', {
      domain:        this.domain,
      mutationCount: this.mutationCount,
      type:          mutationType,
      architecture:  [...this.network.architecture],
    });
  }

  // ── Consolidation ─────────────────────────────────────────────────────────

  /**
   * Once a lesson is learned the region's plasticity drops, making it stable.
   * A plasticity of 0.1 means the effective LR is 10× lower than at birth.
   */
  _consolidate() {
    this.plasticity = Math.max(0.1, 1 - this.accuracy);
    this.emit('consolidated', {
      domain:      this.domain,
      plasticity:  this.plasticity,
      accuracy:    this.accuracy,
    });
  }

  // ── Inference ─────────────────────────────────────────────────────────────

  /**
   * Return the continuous output of this region for the given raw input.
   * Input normalisation and output denormalisation are applied automatically
   * when lesson.normalise is configured, so callers always work in raw units.
   */
  predict(input) {
    const normInput  = this._normalizeInput(input);
    const normOutput = this.network.predict(normInput);
    return this._denormalizeOutput(normOutput);
  }

  /**
   * Return a thresholded (0/1) binary output.  Input normalisation is applied;
   * the binary output itself is never denormalised.
   */
  predictBinary(input, threshold = 0.5) {
    const normInput = this._normalizeInput(input);
    return this.network.predictBinary(normInput, threshold);
  }

  // ── Introspection ─────────────────────────────────────────────────────────

  getInfo() {
    return {
      domain:             this.domain,
      trained:            this.trained,
      plasticity:         this.plasticity,
      accuracy:           this.accuracy,
      mutationCount:      this.mutationCount,
      architecture:       [...this.network.architecture],
      trainingRounds:     this.trainingHistory.length,
    };
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      domain:          this.domain,
      trained:         this.trained,
      plasticity:      this.plasticity,
      accuracy:        this.accuracy,
      mutationCount:   this.mutationCount,
      config:          this.config,
      lesson:          this.lesson,
      trainingHistory: this.trainingHistory,
      network:         this.network.toJSON(),
    };
  }

  static fromJSON(data) {
    const region = new BrainRegion({
      domain:  data.domain,
      lesson:  data.lesson,
      config:  data.config,
    });
    region.trained         = data.trained;
    region.plasticity      = data.plasticity;
    region.accuracy        = data.accuracy;
    region.mutationCount   = data.mutationCount;
    region.trainingHistory = data.trainingHistory || [];
    region.network         = NeuralNetwork.fromJSON(data.network);
    return region;
  }
}

module.exports = BrainRegion;
