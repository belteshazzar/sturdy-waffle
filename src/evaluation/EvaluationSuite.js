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

  evaluateGeneralization({ lessons = [] } = {}) {
    const reports = this._currentPerformance(lessons);
    const accuracies = reports.map(r => r.accuracy).filter(v => typeof v === 'number');
    const gaps = reports.map(r => r.generalizationGap).filter(v => typeof v === 'number');
    const mean = arr => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
    return {
      averageAccuracy: mean(accuracies),
      averageGap: mean(gaps),
      results: reports,
    };
  }

  evaluateLongHorizon({ expressions = [] } = {}) {
    if (!expressions.length) {
      return { evaluated: 0, accuracy: null, solvedRate: null, averageSteps: null, results: [] };
    }
    let correct = 0;
    let solved = 0;
    let totalSteps = 0;
    const results = expressions.map(({ expression, expected }) => {
      try {
        const solvedResult = this.brain.solveString(expression);
        const passed = solvedResult.solved && solvedResult.answer === expected;
        if (passed) correct++;
        if (solvedResult.solved) solved++;
        totalSteps += solvedResult.steps;
        return {
          expression,
          expected,
          actual: solvedResult.answer,
          solved: solvedResult.solved,
          steps: solvedResult.steps,
          passed,
        };
      } catch (err) {
        return {
          expression,
          expected,
          actual: null,
          solved: false,
          steps: 0,
          passed: false,
          error: err.message,
        };
      }
    });
    return {
      evaluated: expressions.length,
      accuracy: correct / expressions.length,
      solvedRate: solved / expressions.length,
      averageSteps: totalSteps / expressions.length,
      results,
    };
  }

  evaluateOOD({ pairs = [], shots = 4 } = {}) {
    if (!pairs.length) return { evaluated: 0, accuracy: null, results: [] };
    const results = [];
    let correct = 0;
    let total = 0;
    for (const pair of pairs) {
      const brain = this.brainFactory();
      brain.learn(pair.source);
      const adapted = this._trainFewShot(brain, pair.source, shots);
      const region = brain.router.route(pair.source.domain);
      const evalData = pair.ood.validationData || pair.ood.trainingData;
      const accuracy = region.evaluateAccuracy(evalData);
      total++;
      if (accuracy >= 0.5) correct++;
      results.push({
        source: pair.source.domain,
        ood: pair.ood.domain,
        adaptedAccuracy: adapted.validationAccuracy,
        oodAccuracy: accuracy,
      });
    }
    return {
      evaluated: total,
      accuracy: total === 0 ? null : correct / total,
      results,
    };
  }

  evaluateSampleEfficiency({ lessons = [], shots = [1, 4, 8] } = {}) {
    const results = shots.map(shotCount => ({
      shots: shotCount,
      results: this._fewShotPerformance(lessons, shotCount),
    }));
    const summary = results.map(entry => {
      const accuracies = entry.results.map(r => r.validationAccuracy).filter(v => typeof v === 'number');
      const avg = accuracies.length ? accuracies.reduce((a, b) => a + b, 0) / accuracies.length : null;
      return { shots: entry.shots, averageAccuracy: avg };
    });
    return { summary, results };
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

  evaluateRelations({ factBase } = {}) {
    if (!factBase) return { evaluated: 0, accuracy: null };
    const relationFacts = factBase.getRelationFacts();
    if (relationFacts.length === 0) return { evaluated: 0, accuracy: null };
    let correct = 0;
    relationFacts.forEach(rel => {
      const actual = this.brain.evaluate({ relation: { name: rel.relation, args: rel.args } });
      if (actual === rel.value) correct++;
    });
    return {
      evaluated: relationFacts.length,
      accuracy: correct / relationFacts.length,
    };
  }

  runAll({ syllabi, expressions, transferPairs, factBase, shots } = {}) {
    return {
      baseline:      this.baseline({ syllabi, shots }),
      compositional: this.evaluateCompositional({ expressions }),
      generalization: this.evaluateGeneralization({ lessons: syllabi ? syllabi.flatMap(s => s.lessons || []) : [] }),
      longHorizon: this.evaluateLongHorizon({ expressions }),
      transfer:      this.evaluateTransfer({ pairs: transferPairs || [], shots }),
      sampleEfficiency: this.evaluateSampleEfficiency({ lessons: syllabi ? syllabi.flatMap(s => s.lessons || []) : [] }),
      selfDiscovery: this.evaluateSelfDiscovery({ factBase }),
      relations:     this.evaluateRelations({ factBase }),
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
