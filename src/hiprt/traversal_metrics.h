#pragma once

#include <cstdint>

// ============================================================================
// Traversal Divergence Metrics
// ============================================================================
//
// Extended divergence metrics for HIPRT traversal profiling.
// These metrics capture divergence at various points during ray tracing.
//
// ============================================================================

struct TraversalMetrics {
    // BVH traversal divergence
    uint32_t traversal_steps;           // Total BVH nodes visited
    uint32_t node_divergence;           // Divergence at internal node decisions

    // Primitive intersection divergence
    uint32_t triangle_tests;            // Number of triangle intersection tests
    uint32_t triangle_divergence;       // Divergence at triangle hit/miss
    uint32_t neural_tests;              // Number of neural AABB intersection tests
    uint32_t neural_divergence;         // Divergence at neural AABB hit/miss

    // Neural inference divergence
    uint32_t early_reject_divergence;   // Divergence at visibility-based early rejection
    uint32_t hash_divergence;           // Divergence in hash encoding (direct vs hash lookup)
    uint32_t mlp_divergence;            // Divergence in MLP (hidden vs output layer)

    // Shadow ray divergence
    uint32_t shadow_tests;              // Number of shadow ray tests
    uint32_t shadow_divergence;         // Divergence at shadow hit/miss

    // Spatial coherence
    float instance_entropy;             // Warp-level instance distribution entropy
};

// Initialize all metrics to zero
inline TraversalMetrics make_zero_metrics() {
    TraversalMetrics m = {};
    return m;
}
