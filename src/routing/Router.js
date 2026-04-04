'use strict';

/**
 * A single node in the routing tree.
 */
class RouteNode {
  constructor(segment) {
    this.segment  = segment;
    this.children = new Map();
    this.region   = null;      // BrainRegion attached at this node (if any)
  }
}

/**
 * A hierarchical domain router that maps dot-notation domain keys to
 * BrainRegion instances.
 *
 * Domain keys are split on '.' to build a tree, e.g.:
 *   "boolean.AND"  → root → boolean → AND
 *   "boolean.OR"   → root → boolean → OR
 *
 * Routing is exact-match first, then prefix-match (deepest matching node).
 * Wildcards ('*') may also be registered.
 *
 * Designed to support future expansion: once there are many domains the tree
 * structure makes it easy to group and selectively query related regions.
 */
class Router {
  constructor() {
    this.root    = new RouteNode('*');
    this.regions = new Map();   // domain string → BrainRegion
  }

  // ── Registration ──────────────────────────────────────────────────────────

  /**
   * Register a BrainRegion under the given domain key.
   * @param {string}      domain
   * @param {BrainRegion} region
   */
  register(domain, region) {
    this.regions.set(domain, region);

    const parts = domain.split('.');
    let node = this.root;
    for (const part of parts) {
      if (!node.children.has(part)) {
        node.children.set(part, new RouteNode(part));
      }
      node = node.children.get(part);
    }
    node.region = region;
  }

  /**
   * Remove a region from the router (tree nodes are retained for structure).
   * @param {string} domain
   */
  unregister(domain) {
    this.regions.delete(domain);
    const parts = domain.split('.');
    let node = this.root;
    for (const part of parts) {
      if (!node.children.has(part)) return;
      node = node.children.get(part);
    }
    node.region = null;
  }

  // ── Routing ───────────────────────────────────────────────────────────────

  /**
   * Find the BrainRegion responsible for the given domain.
   * Returns null when no matching route exists.
   * @param {string} domain
   * @returns {BrainRegion|null}
   */
  route(domain) {
    // Exact match (fast path)
    if (this.regions.has(domain)) return this.regions.get(domain);

    // Prefix / wildcard walk
    const parts = domain.split('.');
    let node       = this.root;
    let lastRegion = null;

    for (const part of parts) {
      if (node.children.has(part)) {
        node = node.children.get(part);
      } else if (node.children.has('*')) {
        node = node.children.get('*');
      } else {
        break;
      }
      if (node.region) lastRegion = node.region;
    }

    return lastRegion;
  }

  // ── Introspection ─────────────────────────────────────────────────────────

  hasRoute(domain) {
    return this.regions.has(domain);
  }

  listDomains() {
    return Array.from(this.regions.keys());
  }

  /**
   * Return the routing tree as a plain object for serialisation / display.
   */
  getTreeStructure() {
    const build = node => ({
      segment:  node.segment,
      domain:   node.region ? node.region.domain : null,
      children: Object.fromEntries(
        Array.from(node.children.entries()).map(([k, v]) => [k, build(v)])
      ),
    });
    return build(this.root);
  }
}

module.exports = Router;
