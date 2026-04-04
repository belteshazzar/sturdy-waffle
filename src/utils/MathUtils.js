'use strict';

/**
 * Xavier/Glorot weight initialisation.
 * Scales random values to avoid vanishing/exploding gradients at startup.
 */
function xavierInit(inputSize, outputSize) {
  const scale = Math.sqrt(2.0 / (inputSize + outputSize));
  return (Math.random() * 2 - 1) * scale;
}

/**
 * Create a 2-D matrix (array of arrays) with an optional per-element init fn.
 */
function createMatrix(rows, cols, initFn = () => 0) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, initFn)
  );
}

/**
 * Create a 1-D vector with an optional per-element init fn.
 */
function createVector(size, initFn = () => 0) {
  return Array.from({ length: size }, initFn);
}

/**
 * Matrix-vector product: matrix [rows x cols] * vec [cols] → result [rows]
 */
function matVecMul(matrix, vec) {
  return matrix.map(row =>
    row.reduce((sum, w, j) => sum + w * vec[j], 0)
  );
}

/**
 * Element-wise vector addition.
 */
function vecAdd(a, b) {
  return a.map((v, i) => v + b[i]);
}

module.exports = {
  xavierInit,
  createMatrix,
  createVector,
  matVecMul,
  vecAdd,
};
