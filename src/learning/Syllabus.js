'use strict';

/**
 * A Syllabus is an ordered collection of Lessons that the Brain works through
 * sequentially.  Ordering matters: simpler concepts should precede complex ones.
 */
class Syllabus {
  /**
   * @param {object}   opts
   * @param {string}   opts.name
   * @param {string}   [opts.description]
   * @param {Lesson[]} [opts.lessons]
   * @param {string[]} [opts.tags]
   */
  constructor({ name, description = '', lessons = [], tags = [] }) {
    if (!name) throw new Error('Syllabus must have a name');
    this.name        = name;
    this.description = description;
    this.lessons     = [...lessons];
    this.tags        = tags;
  }

  /** Append a lesson and return this for chaining. */
  addLesson(lesson) {
    this.lessons.push(lesson);
    return this;
  }

  /** Find a lesson by its domain key. */
  getLessonByDomain(domain) {
    return this.lessons.find(l => l.domain === domain) || null;
  }

  /** Number of lessons in this syllabus. */
  get length() {
    return this.lessons.length;
  }
}

module.exports = Syllabus;
