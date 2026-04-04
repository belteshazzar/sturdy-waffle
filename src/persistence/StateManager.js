'use strict';

const fs   = require('fs');
const path = require('path');

/**
 * StateManager handles persistence of the Brain's complete state to/from JSON.
 *
 * It is intentionally kept as a thin I/O layer so that Brain can be
 * reconstructed without knowing anything about the file system.
 *
 * Usage:
 *   const { saved } = StateManager.save(brain, './saves/my-brain.json');
 *   const brain     = StateManager.load('./saves/my-brain.json');
 */
class StateManager {
  /**
   * Serialise `brain` to a JSON file.  Parent directories are created
   * automatically if they do not exist.
   *
   * @param {Brain}  brain    The Brain instance to serialise
   * @param {string} filepath Destination path (absolute or relative)
   * @returns {{ saved: boolean, filepath: string, sizeBytes: number, regionCount: number }}
   */
  static save(brain, filepath) {
    const dir = path.dirname(filepath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const json = JSON.stringify(brain.toJSON(), null, 2);
    fs.writeFileSync(filepath, json, 'utf8');

    return {
      saved:       true,
      filepath,
      sizeBytes:   Buffer.byteLength(json, 'utf8'),
      regionCount: brain.regions.size,
    };
  }

  /**
   * Load a Brain from a previously saved JSON file.
   * Requires the Brain class; loaded lazily to avoid circular dependency.
   *
   * @param {string} filepath
   * @returns {Brain}
   */
  static load(filepath) {
    if (!fs.existsSync(filepath)) {
      throw new Error(`Brain state file not found: ${filepath}`);
    }
    const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
    // Lazy-require Brain to avoid a circular module dependency
    const Brain = require('../brain/Brain');
    return Brain.fromJSON(data);
  }

  /**
   * Read the raw JSON object from a saved state file without reconstructing
   * the Brain.  Useful for tooling/inspection.
   *
   * @param {string} filepath
   * @returns {object}
   */
  static loadRaw(filepath) {
    if (!fs.existsSync(filepath)) {
      throw new Error(`Brain state file not found: ${filepath}`);
    }
    return JSON.parse(fs.readFileSync(filepath, 'utf8'));
  }
}

module.exports = StateManager;
