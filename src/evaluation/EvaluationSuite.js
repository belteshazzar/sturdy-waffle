'use strict';

const Lesson = require('../learning/Lesson');

class EvaluationSuite {
  constructor({ brain, brainFactory }) {
    this.brain = brain;
    this.brainFactory = brainFactory;
  }

  baseline({ syllabi = [], shots = 4 } = {}) {
    const lessons = syllabi.flatMap(s => s.lessons || []);
    const inventory = this._inventory();
    const currentPerformance = this._currentPerformance(lessons);
    const fewShot = this._fewShotPerformance(lessons, shots);
    const forgetting = this._forgettingPerformance(lessons, shots);

    return {
      inventory,
      currentPerformance,
      fewShot,
      forgetting,
    };
  }

  evaluateCompositional({ expressions = [] } = {}) {
    if (!expressions.length) {
      return { evaluated: 0, accuracy: null, results: [] };
    }
    let correct = 0;
    const results = expressions.map(({ expression, expected }) => {
      const actual = this.brain.evaluate(expression);
      const passed = actual === expected;
      if (passed) correct++;
      return { expression, expected, actual, passed };
    });
    return {
      evaluated: expressions.length,
      accuracy:  correct / expressions.length,
      results,
    };
  }

  evaluateTransfer({ pairs = [], shots = 4 } = {}) {
    const results = [];
    for (const pair of pairs) {
      const brain = this.brainFactory();
      brain.learn(pair.source);
      const adapted = this._trainFewShot(brain, pair.target, shots);
      results.push({
        source: pair.source.domain,
        target: pair.target.domain,
        accuracy: adapted.validationAccuracy,
      });
    }
    return results;
  }

  evaluateSelfDiscovery({ factBase, promote = false, minSupport = 2 } = {}) {
    if (!factBase) return { discovered: 0, validated: 0, precision: null };
    const discoveries = this.brain.selfLearn({ promoteToFactBase: promote, minSupport });
    const validated = discoveries?.facts?.filter(f => factBase.has(f.subject, f.predicate)).length || 0;
    const discovered = discoveries?.facts?.length || 0;
    return {
      discovered,
      validated,
      precision: discovered === 0 ? null : validated / discovered,
    };
  }

  runAll({ syllabi, expressions, transferPairs, factBase, shots } = {}) {
    return {
      baseline:      this.baseline({ syllabi, shots }),
      compositional: this.evaluateCompositional({ expressions }),
      transfer:      this.evaluateTransfer({ pairs: transferPairs || [], shots }),
      selfDiscovery: this.evaluateSelfDiscovery({ factBase }),
    };
  }

  _inventory() {
    const domains = Array.from(this.brain.regions.keys());
    const categories = {};
    domains.forEach(domain => {
      const category = domain.split('.')[0];
      if (!categories[category]) categories[category] = [];
      categories[category].push(domain);
    });
    return {
      domains,
      categories,
      regionCount: domains.length,
    };
  }

  _currentPerformance(lessons) {
    const results = [];
    for (const lesson of lessons) {
      const region = this.brain.router.route(lesson.domain);
      if (!region) continue;
      const validationData = lesson.validationData || lesson.trainingData;
      const accuracy = region.evaluateAccuracy(validationData);
      const trainingAccuracy = region.evaluateAccuracy(lesson.trainingData);
      results.push({
        domain: lesson.domain,
        accuracy,
        trainingAccuracy,
        generalizationGap: trainingAccuracy - accuracy,
      });
    }
    return results;
  }

  _fewShotPerformance(lessons, shots) {
    const results = [];
    for (const lesson of lessons) {
      const brain = this.brainFactory();
      const report = this._trainFewShot(brain, lesson, shots);
      results.push(report);
    }
    return results;
  }

  _forgettingPerformance(lessons, shots) {
    if (!lessons.length) return [];
    const brain = this.brainFactory();
    const results = [];
    const trained = [];

    for (const lesson of lessons) {
      const report = this._trainFewShot(brain, lesson, shots);
      trained.push(lesson);
      const retention = trained.map(prevLesson => {
        const region = brain.router.route(prevLesson.domain);
        if (!region) return null;
        const accuracy = region.evaluateAccuracy(prevLesson.validationData || prevLesson.trainingData);
        return { domain: prevLesson.domain, accuracy };
      }).filter(Boolean);
      results.push({
        learned: lesson.domain,
        retention,
      });
    }
    return results;
  }

  _trainFewShot(brain, lesson, shots) {
    const trainingData = lesson.trainingData.slice(0, shots);
    const smallLesson = new Lesson({
      name:          `${lesson.name} (few-shot)`,
      domain:        lesson.domain,
      description:   lesson.description,
      trainingData,
      validationData: lesson.validationData || lesson.trainingData,
      inputSize:     lesson.inputSize,
      outputSize:    lesson.outputSize,
      tags:          lesson.tags,
      networkConfig: lesson.networkConfig,
      mode:          lesson.mode,
      normalise:     lesson.normalise,
    });
    brain.learn(smallLesson);
    const region = brain.router.route(lesson.domain);
    const validationData = lesson.validationData || lesson.trainingData;
    const validationAccuracy = region.evaluateAccuracy(validationData);
    const trainingAccuracy = region.evaluateAccuracy(trainingData);
    return {
      domain: lesson.domain,
      shots,
      validationAccuracy,
      trainingAccuracy,
      generalizationGap: trainingAccuracy - validationAccuracy,
    };
  }
}

module.exports = EvaluationSuite;
