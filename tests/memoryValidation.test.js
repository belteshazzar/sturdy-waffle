'use strict';

const MemorySystem = require('../src/memory/MemorySystem');

describe('MemorySystem validation', () => {
  test('detects corrupted episodic entries', () => {
    const memory = new MemorySystem();
    memory.episodic.episodes.push({ input: 'bad', output: [] });
    const report = memory.validate();
    expect(report.valid).toBe(false);
    expect(report.issues.length).toBeGreaterThan(0);
  });

  test('passes valid episodes', () => {
    const memory = new MemorySystem();
    memory.episodic.addEpisode({ domain: 'test', input: [1], output: [0] });
    const report = memory.validate({ strict: true });
    expect(report.valid).toBe(true);
  });
});
