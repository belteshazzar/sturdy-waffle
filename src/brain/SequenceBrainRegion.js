'use strict';

const { EventEmitter } = require('events');
const RecurrentNetwork = require('./RecurrentNetwork');

class SequenceBrainRegion extends EventEmitter {
  constructor({ domain, lesson, config = {} }) {
    super();
    this.domain          = domain;
    this.trained         = false;
    this.plasticity      = 1.0;
    this.accuracy        = 0;
    this.trainingHistory = [];
    this.mutationCount   = 0;
    this.lesson          = lesson;

    this.config = {
      targetAccuracy:           0.98,
      maxMutations:             6,
      epochsPerRound:           200,
      maxEpochsTotal:           8000,
      accuracyPlateauThreshold: 0.005,
      mutationPatience:         3,
      baseLearningRate:         0.05,
      regressionTolerance:      0.08,
      ...config,
    };

    this._initializeNetwork(lesson);
  }

  _inferStepSize(lesson) {
    if (lesson.inputSize) return lesson.inputSize;
    const sample = lesson.trainingData && lesson.trainingData[0];
    if (!sample) return 1;
    const seq = Array.isArray(sample.input) ? sample.input : [];
    if (Array.isArray(seq[0])) return seq[0].length;
    return 1;
  }

  _initializeNetwork(lesson) {
    const inputSize = this._inferStepSize(lesson);
    const outputSize = lesson.outputSize || (lesson.trainingData[0]?.output?.length || 1);
    const hiddenSize = lesson.sequenceHiddenSize || Math.max(8, inputSize * 4);
    const netConfig = lesson.networkConfig || {};
    const outputActivation = netConfig.outputActivation || (lesson.mode === 'multiclass' ? 'sigmoid' : 'linear');

    this.network = new RecurrentNetwork({
      inputSize,
      hiddenSize,
      outputSize,
      learningRate: this.config.baseLearningRate,
      hiddenActivation: netConfig.hiddenActivation || 'tanh',
      outputActivation,
    });

    this._trainingData = lesson.trainingData.map(sample => this._prepareSample(sample));
    const rawVal = lesson.validationData || lesson.trainingData;
    this._validationData = rawVal.map(sample => this._prepareSample(sample));
  }

  _prepareSequence(input) {
    if (Array.isArray(input) && Array.isArray(input[0])) return input;
    if (Array.isArray(input)) return input.map(v => [v]);
    return [[input]];
  }

  _prepareSample(sample) {
    return {
      input: this._prepareSequence(sample.input),
      output: sample.output,
    };
  }

  _measureAccuracy(data) {
    if (this.lesson.mode === 'regression') {
      return this.network.regressionAccuracy(data, this.config.regressionTolerance);
    }
    if (this.lesson.mode === 'multiclass') {
      return this.network.multiclassAccuracy(data);
    }
    if (this.lesson.mode === 'multilabel') {
      return this.network.hammingAccuracy(data);
    }
    return this.network.accuracy(data);
  }

  train() {
    if (this.trained) {
      this.accuracy = this._measureAccuracy(this._validationData);
      if (this.accuracy >= this.config.targetAccuracy) {
        return { trained: true, accuracy: this.accuracy, mutationCount: this.mutationCount, totalEpochs: 0 };
      }
      this.trained = false;
    }

    let bestAccuracy = 0;
    let roundsWithoutImprovement = 0;
    let totalEpochs = 0;

    this.emit('training:start', { domain: this.domain, architecture: [...this.network.architecture] });

    while (totalEpochs < this.config.maxEpochsTotal) {
      const effectiveLR = Math.max(0.001, this.config.baseLearningRate * this.plasticity);
      this.network.learningRate = effectiveLR;

      const { finalLoss } = this.network.train(this._trainingData, this.config.epochsPerRound);
      totalEpochs += this.config.epochsPerRound;

      this.accuracy = this._measureAccuracy(this._validationData);

      const snapshot = {
        epoch: totalEpochs,
        loss: finalLoss,
        accuracy: this.accuracy,
        architecture: [...this.network.architecture],
        plasticity: this.plasticity,
      };
      this.trainingHistory.push(snapshot);
      this.emit('training:progress', { domain: this.domain, ...snapshot });

      if (this.accuracy >= this.config.targetAccuracy) {
        this.trained = true;
        this._consolidate();
        break;
      }

      if (this.accuracy > bestAccuracy + this.config.accuracyPlateauThreshold) {
        bestAccuracy = this.accuracy;
        roundsWithoutImprovement = 0;
      } else {
        roundsWithoutImprovement++;
      }

      if (roundsWithoutImprovement >= this.config.mutationPatience && this.mutationCount < this.config.maxMutations) {
        this._mutate();
        roundsWithoutImprovement = 0;
        bestAccuracy = 0;
      }
    }

    if (this.trained) {
      this.emit('region:trained', {
        domain: this.domain,
        trained: this.trained,
        accuracy: this.accuracy,
        mutationCount: this.mutationCount,
        totalEpochs,
        architecture: [...this.network.architecture],
      });
    }

    return { trained: this.trained, accuracy: this.accuracy, mutationCount: this.mutationCount, totalEpochs };
  }

  _mutate() {
    this.mutationCount++;
    const newHidden = Math.min(this.network.hiddenSize + 4, 64);
    this.network = new RecurrentNetwork({
      inputSize: this.network.inputSize,
      hiddenSize: newHidden,
      outputSize: this.network.outputSize,
      learningRate: this.config.baseLearningRate,
      hiddenActivation: this.network.hiddenActivation,
      outputActivation: this.network.outputActivation,
    });
    this.emit('mutation', {
      domain: this.domain,
      mutationCount: this.mutationCount,
      type: `expandHidden(${newHidden})`,
      architecture: [...this.network.architecture],
    });
  }

  _consolidate() {
    this.plasticity = Math.max(0.1, 1 - this.accuracy);
    this.emit('consolidated', {
      domain: this.domain,
      plasticity: this.plasticity,
      accuracy: this.accuracy,
    });
  }

  predict(input) {
    const seq = this._prepareSequence(input);
    return this.network.predict(seq);
  }

  predictBinary(input, threshold = 0.5) {
    const seq = this._prepareSequence(input);
    return this.network.predictBinary(seq, threshold);
  }

  predictArgmax(input) {
    const seq = this._prepareSequence(input);
    return this.network.predictArgmax(seq);
  }

  getInfo() {
    return {
      domain: this.domain,
      trained: this.trained,
      plasticity: this.plasticity,
      accuracy: this.accuracy,
      mutationCount: this.mutationCount,
      architecture: [...this.network.architecture],
      trainingRounds: this.trainingHistory.length,
      sequence: true,
    };
  }

  evaluateAccuracy(rawSamples) {
    const samples = rawSamples.map(sample => this._prepareSample(sample));
    return this._measureAccuracy(samples);
  }

  toJSON() {
    return {
      type: 'sequence',
      domain: this.domain,
      trained: this.trained,
      plasticity: this.plasticity,
      accuracy: this.accuracy,
      mutationCount: this.mutationCount,
      config: this.config,
      lesson: this.lesson,
      trainingHistory: this.trainingHistory,
      network: this.network.toJSON(),
    };
  }

  static fromJSON(data) {
    const region = new SequenceBrainRegion({
      domain: data.domain,
      lesson: data.lesson,
      config: data.config,
    });
    region.trained = data.trained;
    region.plasticity = data.plasticity;
    region.accuracy = data.accuracy;
    region.mutationCount = data.mutationCount;
    region.trainingHistory = data.trainingHistory || [];
    region.network = RecurrentNetwork.fromJSON(data.network);
    return region;
  }
}

module.exports = SequenceBrainRegion;
