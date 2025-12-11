#pragma once

#include <cuda_runtime.h>

// ============================================================================
// Warp Divergence Profiling Utilities
// ============================================================================
//
// This file provides utilities for measuring warp divergence in OptiX programs
// using __ballot_sync() and __popc() intrinsics.
//
// These measurements help identify divergence hotspots without requiring
// NSight Compute access.
//
// ============================================================================

// Measure divergence at a branch point
// Returns the number of threads that diverge (take the minority path)
// This represents "wasted work" - threads that must idle during the branch
__device__ __forceinline__ uint32_t measure_divergence(bool condition) {
    // Get active mask for current warp (which threads are active right now)
    // unsigned int active_mask = __activemask();

    // // Ballot: which threads in this warp satisfy the condition?
    // // Each bit in the result represents one thread's vote
    // unsigned int ballot = __ballot_sync(active_mask, condition);

    // // Count threads taking the TRUE path
    // unsigned int true_count = __popc(ballot);

    // // Count threads taking the FALSE path
    // unsigned int false_count = __popc(active_mask & ~ballot);

    // // Divergence metric: min(true_count, false_count)
    // // This measures "wasted work" - threads that idle during divergence
    // // For example:
    // //   - If 28 threads take TRUE and 4 take FALSE, divergence = 4
    // //   - If 16 threads take TRUE and 16 take FALSE, divergence = 16 (worst case)
    // //   - If all threads take same path, divergence = 0 (best case)
    // return min(true_count, false_count);
    return 0;
}

// Accumulate divergence into a counter
// Usage: record_divergence(counter, some_condition);
__device__ __forceinline__ void record_divergence(
    unsigned int& counter,
    bool condition
) {
    uint32_t divergence = measure_divergence(condition);
    counter += divergence;
}

// Count number of unique instances in warp (simplified for OptiX compatibility)
// Measures spatial coherence - how scattered rays are across instances
// instanceID: this thread's instance ID (-1 for miss, 0+ for hit)
// Returns: approximate unique instance count (not exact entropy, but correlates)
//   - Returns min(instance_id_this_thread, 16) as a proxy for instance mixing
//   - Higher values = more scattered across instances
static __forceinline__ __device__
float warpInstanceEntropy(int instanceID)
{
    // Active lanes in this warp
    unsigned mask = __activemask();
    int lane      = threadIdx.x & 31;

    // Number of active lanes (may be < 32 at edges)
    int activeCount = __popc(mask);
    if (activeCount <= 1)
        return 0.0f; // trivially coherent

    float entropy = 0.0f;

    // Loop over all possible lanes, but only consider ones that are active
    for (int leader = 0; leader < 32; ++leader)
    {
        if ((mask & (1u << leader)) == 0)
            continue; // this lane isn't active in this warp

        // Broadcast the instanceID from 'leader' to all lanes
        int leaderID = __shfl_sync(__activemask(), instanceID, leader);

        // Build a mask of all lanes whose instanceID matches leaderID
        unsigned groupMask = __ballot_sync(__activemask(), instanceID == leaderID);

        // Find the lowest lane index in this group (the canonical leader)
        int groupLeader = __ffs(groupMask) - 1;

        // Only the canonical leader for each distinct group contributes
        if (lane == leader && leader == groupLeader)
        {
            int groupSize = __popc(groupMask);
            float p = float(groupSize) / float(activeCount);
            entropy += -p * __log2f(p);
        }
    }

    // Broadcast final entropy from the first active lane to everyone
    int firstLane = __ffs(mask) - 1;
    entropy = __shfl_sync(__activemask(), entropy, firstLane);
    return entropy;
}

