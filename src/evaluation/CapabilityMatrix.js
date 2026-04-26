'use strict';

const DEFAULT_TARGETS = Object.freeze({
  perception:    { maxTokens: 512, maxLines: 200 },
  representation:{ regionAccuracy: 0.9 },
  reasoning:     { compositionalAccuracy: 0.9 },
  memory:        { retention: 0.85 },
  learning:      { sampleEfficiency: 0.75 },
  planning:      { horizon: 5 },
});

const MODULE_CAPABILITIES = Object.freeze({
  brain:         ['representation', 'reasoning', 'learning', 'planning'],
  decomposition: ['reasoning', 'planning'],
  memory:        ['memory'],
  worldModel:    ['planning', 'representation'],
  parsing:       ['perception'],
});

function buildCapabilityMatrix({ brain, targets = {} } = {}) {
  const resolvedTargets = {
    ...DEFAULT_TARGETS,
    ...targets,
  };

  const regionInfos = brain ? Array.from(brain.regions.values()).map(r => r.getInfo()) : [];
  const regionAccuracies = regionInfos.map(info => info.accuracy).filter(v => typeof v === 'number');
  const averageAccuracy = regionAccuracies.length
    ? regionAccuracies.reduce((acc, v) => acc + v, 0) / regionAccuracies.length
    : null;

  const memoryInfo = brain?.memory?.getInfo?.() || null;
  const worldInfo = brain?.worldModel?.getInfo?.() || null;

  const capabilityStatus = {
    perception: {
      targets: resolvedTargets.perception,
      current: {
        supportedInputs: ['expression', 'tokens', 'knowledge', 'query'],
        maxTokens: brain?.config?.inputLimits?.maxTokens ?? null,
        maxLines: brain?.config?.inputLimits?.maxLines ?? null,
      },
    },
    representation: {
      targets: resolvedTargets.representation,
      current: {
        regionCount: brain?.regions?.size ?? 0,
        averageAccuracy,
        sharedEmbedding: !!brain?.sharedEmbedding,
      },
    },
    reasoning: {
      targets: resolvedTargets.reasoning,
      current: {
        compositionalAccuracy: averageAccuracy,
        controllerEnabled: !!brain?.controller,
        learnedRouterEnabled: !!brain?.learnedRouter,
      },
    },
    memory: {
      targets: resolvedTargets.memory,
      current: {
        episodic: memoryInfo?.episodic ?? null,
        semantic: memoryInfo?.semantic ?? null,
        lastConsolidation: memoryInfo?.consolidation ?? null,
      },
    },
    learning: {
      targets: resolvedTargets.learning,
      current: {
        averageAccuracy,
        metaLearningEnabled: !!brain?.metaLearner,
        selfSupervisedEnabled: !!brain?.selfSupervisedLearner,
      },
    },
    planning: {
      targets: resolvedTargets.planning,
      current: {
        worldModelEnabled: !!brain?.worldModel,
        worldModel: worldInfo,
        controllerEnabled: !!brain?.controller,
      },
    },
  };

  const gaps = Object.fromEntries(
    Object.entries(capabilityStatus).map(([capability, data]) => {
      const target = data.targets;
      const current = data.current;
      if (target && typeof target.regionAccuracy === 'number' && typeof current.averageAccuracy === 'number') {
        return [capability, Math.max(0, target.regionAccuracy - current.averageAccuracy)];
      }
      if (target && typeof target.compositionalAccuracy === 'number' &&
          typeof current.compositionalAccuracy === 'number') {
        return [capability, Math.max(0, target.compositionalAccuracy - current.compositionalAccuracy)];
      }
      return [capability, null];
    })
  );

  return {
    targets: resolvedTargets,
    modules: MODULE_CAPABILITIES,
    capabilities: capabilityStatus,
    gaps,
  };
}

module.exports = {
  DEFAULT_TARGETS,
  MODULE_CAPABILITIES,
  buildCapabilityMatrix,
};
