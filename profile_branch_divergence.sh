#!/bin/bash
# Branch Divergence Profiling Script
# Profiles mesh and neural scenes across different sphere counts
# and compares branch divergence metrics

set -e

# Configuration
SPHERE_COUNTS=(1 2 4 8 16 32 64)
SCENE_TYPES=("mesh" "neural")
OUTPUT_DIR="output"
RESULTS_FILE="branch_divergence_results.csv"
ANALYZE_SCRIPT="scripts/analyze_divergence.py"

# Parse command-line arguments
NO_SHADOWS=false
DEBUG_BUILD=false
for arg in "$@"; do
    case $arg in
        --no-shadows|-S)
            NO_SHADOWS=true
            shift
            ;;
        --debug|-d)
            DEBUG_BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-shadows, -S   Disable shadow rays during profiling"
            echo "  --debug, -d        Use Debug build with line info for source analysis"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                     # Profile Release build with shadows"
            echo "  $0 --debug             # Profile Debug build for line-level analysis"
            echo "  $0 --no-shadows        # Profile Release build without shadows"
            echo "  $0 --debug --no-shadows # Profile Debug build without shadows"
            exit 0
            ;;
    esac
done

# Set renderer path based on build type
if [ "$DEBUG_BUILD" = true ]; then
    RENDERER="build/bin/Debug/renderer_hiprt"
    BUILD_SUFFIX="_debug"
    NCU_SOURCE_FLAGS="--import-source on --source-folders /opt/hiprt,$(pwd)"
    echo -e "${YELLOW}Using DEBUG build for line-level source analysis${NC}"
else
    RENDERER="build/bin/Release/renderer_hiprt"
    BUILD_SUFFIX=""
    NCU_SOURCE_FLAGS=""
fi

# Suffix for output files when shadows disabled
if [ "$NO_SHADOWS" = true ]; then
    SHADOW_SUFFIX="_no_shadows"
    RENDERER_ARGS="--no-shadows"
    echo "Shadow rays DISABLED for this profiling run"
else
    SHADOW_SUFFIX=""
    RENDERER_ARGS=""
fi

# Combine suffixes for output files
OUTPUT_SUFFIX="${BUILD_SUFFIX}${SHADOW_SUFFIX}"
RESULTS_FILE="branch_divergence_results${OUTPUT_SUFFIX}.csv"

# NSight Compute path
NCU="/usr/local/cuda/bin/ncu"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "       Branch Divergence Profiling for Mesh vs Neural"
echo "============================================================"
echo ""

# Check if renderer exists
if [ ! -f "$RENDERER" ]; then
    echo -e "${RED}Error: Renderer not found at $RENDERER${NC}"
    echo "Please build the project first."
    exit 1
fi

# Initialize results file with header
echo "scene_type,num_spheres,branch_targets,divergent_threads,uniform_threads,divergence_ratio,branch_efficiency,avg_active_threads,warp_efficiency,efficiency_gap,fma_ratio,l1_hit_rate,l2_hit_rate,occupancy,mem_stall,registers,duration_ms" > "$RESULTS_FILE"

# Declare associative arrays to store results
declare -A MESH_DIVERGENCE
declare -A NEURAL_DIVERGENCE
declare -A MESH_TARGETS
declare -A NEURAL_TARGETS
declare -A MESH_DIVERGENT
declare -A NEURAL_DIVERGENT
declare -A MESH_BRANCH_EFF
declare -A NEURAL_BRANCH_EFF
declare -A MESH_AVG_THREADS
declare -A NEURAL_AVG_THREADS
declare -A MESH_WARP_EFF
declare -A NEURAL_WARP_EFF
declare -A MESH_EFF_GAP
declare -A NEURAL_EFF_GAP
declare -A MESH_FFMA
declare -A NEURAL_FFMA
declare -A MESH_TOTAL_INST
declare -A NEURAL_TOTAL_INST
declare -A MESH_L1_HIT
declare -A NEURAL_L1_HIT
declare -A MESH_L2_HIT
declare -A NEURAL_L2_HIT
declare -A MESH_OCCUPANCY
declare -A NEURAL_OCCUPANCY
declare -A MESH_MEM_STALL
declare -A NEURAL_MEM_STALL
declare -A MESH_REGISTERS
declare -A NEURAL_REGISTERS
declare -A MESH_DURATION
declare -A NEURAL_DURATION

# Function to convert metric value (handles K, M suffixes)
convert_value() {
    local val="$1"
    if [[ "$val" == *"K" ]]; then
        echo "$val" | sed 's/K//' | awk '{printf "%.0f", $1 * 1000}'
    elif [[ "$val" == *"M" ]]; then
        echo "$val" | sed 's/M//' | awk '{printf "%.0f", $1 * 1000000}'
    elif [[ "$val" == *"G" ]]; then
        echo "$val" | sed 's/G//' | awk '{printf "%.0f", $1 * 1000000000}'
    else
        echo "$val"
    fi
}

echo "Starting profiling runs..."
echo ""

# Profile each scene
for scene_type in "${SCENE_TYPES[@]}"; do
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Profiling ${scene_type^^} scenes${NC}"
    echo -e "${CYAN}========================================${NC}"

    for num_spheres in "${SPHERE_COUNTS[@]}"; do
        scene_file="scenes/scene_${scene_type}_${num_spheres}.yaml"
        output_base="${OUTPUT_DIR}/${scene_type}_${num_spheres}${OUTPUT_SUFFIX}"
        output_file="${output_base}_render.ppm"

        if [ ! -f "$scene_file" ]; then
            echo -e "${YELLOW}Warning: Scene file not found: $scene_file${NC}"
            continue
        fi

        echo -e "${BLUE}Profiling: ${scene_type} with ${num_spheres} sphere(s)...${NC}"

        # Run NSight Compute profiling
        # Metrics:
        #   - smsp__sass_branch_targets.sum: Total branch targets
        #   - smsp__sass_branch_targets_threads_divergent.sum: Divergent thread-targets
        #   - smsp__sass_branch_targets_threads_uniform.sum: Uniform thread-targets
        #   - smsp__sass_average_branch_targets_threads_uniform.pct: Branch efficiency (%)
        #   - smsp__thread_inst_executed_per_inst_executed.ratio: Avg active threads per instruction
        #   - sm__sass_thread_inst_executed_op_ffma_pred_on.sum: FMA operations (compute intensity)
        #   - sm__sass_thread_inst_executed.sum: Total thread instructions
        #   - l1tex__t_sector_hit_rate.pct: L1 cache hit rate
        #   - lts__t_sector_hit_rate.pct: L2 cache hit rate
        #   - sm__warps_active.avg.pct_of_peak_sustained_active: Occupancy
        #   - smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct: Memory stalls
        #   - launch__registers_per_thread: Registers per thread
        #   - gpu__time_duration.sum: Kernel execution time
        sudo -E "$NCU" \
            --metrics smsp__sass_branch_targets.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_branch_targets_threads_uniform.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,launch__registers_per_thread,gpu__time_duration.sum \
            --set full \
            -o "$output_base" \
            -f \
            --replay-mode=kernel \
            --kernel-name-base=demangled \
            --kernel-name=renderKernel \
            --target-processes=all \
            $NCU_SOURCE_FLAGS \
            "$RENDERER" "$scene_file" "$output_file" $RENDERER_ARGS 2>&1 > /dev/null

        ncu_output=$(ncu --import "${output_base}.ncu-rep" --page details 2>/dev/null || true)

        # Extract metrics - look for the summary/total values
        branch_targets=$(echo "$ncu_output" | grep "smsp__sass_branch_targets.sum" | tail -1 | awk '{print $NF}' | tr -d ',')
        divergent_threads=$(echo "$ncu_output" | grep "smsp__sass_branch_targets_threads_divergent.sum" | tail -1 | awk '{print $NF}' | tr -d ',')
        uniform_threads=$(echo "$ncu_output" | grep "smsp__sass_branch_targets_threads_uniform.sum" | tail -1 | awk '{print $NF}' | tr -d ',')

        # Extract efficiency metrics
        branch_efficiency=$(echo "$ncu_output" | grep "smsp__sass_average_branch_targets_threads_uniform.pct" | tail -1 | awk '{print $NF}' | tr -d ',')
        avg_active_threads=$(echo "$ncu_output" | grep "smsp__thread_inst_executed_per_inst_executed.ratio" | tail -1 | awk '{print $NF}' | tr -d ',')

        # Extract compute intensity metrics
        ffma_ops=$(echo "$ncu_output" | grep "sm__sass_thread_inst_executed_op_ffma_pred_on.sum" | tail -1 | awk '{print $NF}' | tr -d ',')
        total_inst=$(echo "$ncu_output" | grep "sm__sass_thread_inst_executed.sum" | tail -1 | awk '{print $NF}' | tr -d ',')

        # Extract memory metrics
        l1_hit_rate=$(echo "$ncu_output" | grep "l1tex__t_sector_hit_rate.pct" | tail -1 | awk '{print $NF}' | tr -d ',')
        l2_hit_rate=$(echo "$ncu_output" | grep "lts__t_sector_hit_rate.pct" | tail -1 | awk '{print $NF}' | tr -d ',')

        # Extract occupancy and stall metrics
        occupancy=$(echo "$ncu_output" | grep "sm__warps_active.avg.pct_of_peak_sustained_active" | tail -1 | awk '{print $NF}' | tr -d ',')
        mem_stall=$(echo "$ncu_output" | grep "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct" | tail -1 | awk '{print $NF}' | tr -d ',')

        # Extract register and timing metrics
        registers=$(echo "$ncu_output" | grep "launch__registers_per_thread" | tail -1 | awk '{print $NF}' | tr -d ',')
        # Duration comes in nanoseconds, convert to milliseconds
        duration_ns=$(echo "$ncu_output" | grep "gpu__time_duration.sum" | tail -1 | awk '{print $NF}' | tr -d ',')
        if [ -n "$duration_ns" ] && [ "$duration_ns" != "0" ]; then
            duration_ms=$(awk "BEGIN {printf \"%.3f\", $duration_ns / 1000000}")
        else
            duration_ms="N/A"
        fi

        # Convert values (handle K, M, G suffixes)
        branch_targets_num=$(convert_value "$branch_targets")
        divergent_threads_num=$(convert_value "$divergent_threads")
        uniform_threads_num=$(convert_value "$uniform_threads")

        # Calculate divergence ratio
        if [ -n "$branch_targets_num" ] && [ "$branch_targets_num" != "0" ] && [ -n "$divergent_threads_num" ]; then
            divergence_ratio=$(awk "BEGIN {printf \"%.6f\", $divergent_threads_num / $branch_targets_num}")
        else
            divergence_ratio="N/A"
        fi

        # Calculate warp efficiency and efficiency gap
        # Warp efficiency = avg_active_threads / 32 * 100
        # Efficiency gap = branch_efficiency - warp_efficiency
        if [ -n "$avg_active_threads" ] && [ "$avg_active_threads" != "0" ]; then
            warp_efficiency=$(awk "BEGIN {printf \"%.2f\", $avg_active_threads / 32 * 100}")
        else
            warp_efficiency="N/A"
        fi

        if [ -n "$branch_efficiency" ] && [ "$branch_efficiency" != "N/A" ] && [ "$warp_efficiency" != "N/A" ]; then
            efficiency_gap=$(awk "BEGIN {printf \"%.2f\", $branch_efficiency - $warp_efficiency}")
        else
            efficiency_gap="N/A"
        fi

        # Convert new metric values
        ffma_ops_num=$(convert_value "$ffma_ops")
        total_inst_num=$(convert_value "$total_inst")

        # Calculate compute intensity (FMA ratio)
        if [ -n "$total_inst_num" ] && [ "$total_inst_num" != "0" ] && [ -n "$ffma_ops_num" ]; then
            fma_ratio=$(awk "BEGIN {printf \"%.4f\", $ffma_ops_num / $total_inst_num * 100}")
        else
            fma_ratio="N/A"
        fi

        # Store results
        if [ "$scene_type" == "mesh" ]; then
            MESH_DIVERGENCE[$num_spheres]=$divergence_ratio
            MESH_TARGETS[$num_spheres]=$branch_targets_num
            MESH_DIVERGENT[$num_spheres]=$divergent_threads_num
            MESH_BRANCH_EFF[$num_spheres]=$branch_efficiency
            MESH_AVG_THREADS[$num_spheres]=$avg_active_threads
            MESH_WARP_EFF[$num_spheres]=$warp_efficiency
            MESH_EFF_GAP[$num_spheres]=$efficiency_gap
            MESH_FFMA[$num_spheres]=$fma_ratio
            MESH_TOTAL_INST[$num_spheres]=$total_inst_num
            MESH_L1_HIT[$num_spheres]=$l1_hit_rate
            MESH_L2_HIT[$num_spheres]=$l2_hit_rate
            MESH_OCCUPANCY[$num_spheres]=$occupancy
            MESH_MEM_STALL[$num_spheres]=$mem_stall
            MESH_REGISTERS[$num_spheres]=$registers
            MESH_DURATION[$num_spheres]=$duration_ms
        else
            NEURAL_DIVERGENCE[$num_spheres]=$divergence_ratio
            NEURAL_TARGETS[$num_spheres]=$branch_targets_num
            NEURAL_DIVERGENT[$num_spheres]=$divergent_threads_num
            NEURAL_BRANCH_EFF[$num_spheres]=$branch_efficiency
            NEURAL_AVG_THREADS[$num_spheres]=$avg_active_threads
            NEURAL_WARP_EFF[$num_spheres]=$warp_efficiency
            NEURAL_EFF_GAP[$num_spheres]=$efficiency_gap
            NEURAL_FFMA[$num_spheres]=$fma_ratio
            NEURAL_TOTAL_INST[$num_spheres]=$total_inst_num
            NEURAL_L1_HIT[$num_spheres]=$l1_hit_rate
            NEURAL_L2_HIT[$num_spheres]=$l2_hit_rate
            NEURAL_OCCUPANCY[$num_spheres]=$occupancy
            NEURAL_MEM_STALL[$num_spheres]=$mem_stall
            NEURAL_REGISTERS[$num_spheres]=$registers
            NEURAL_DURATION[$num_spheres]=$duration_ms
        fi

        # Write to CSV
        echo "${scene_type},${num_spheres},${branch_targets_num},${divergent_threads_num},${uniform_threads_num},${divergence_ratio},${branch_efficiency},${avg_active_threads},${warp_efficiency},${efficiency_gap},${fma_ratio},${l1_hit_rate},${l2_hit_rate},${occupancy},${mem_stall},${registers},${duration_ms}" >> "$RESULTS_FILE"

        echo "  Branch Efficiency: ${branch_efficiency}%"
        echo "  Warp Efficiency: ${warp_efficiency}% (${avg_active_threads}/32 threads)"
        echo "  Efficiency Gap: ${efficiency_gap}%"
        echo "  FMA Ratio: ${fma_ratio}%"
        echo "  L1 Hit Rate: ${l1_hit_rate}%"
        echo "  L2 Hit Rate: ${l2_hit_rate}%"
        echo "  Occupancy: ${occupancy}%"
        echo "  Memory Stalls: ${mem_stall}%"
        echo "  Registers/Thread: ${registers}"
        echo "  Kernel Duration: ${duration_ms} ms"
        echo ""
    done
done

# Print comparison table
echo ""
echo "============================================================"
echo "                    RESULTS SUMMARY"
echo "============================================================"
echo ""

# Print header with ratio columns
printf "${CYAN}%-8s │ %-12s │ %-12s │ %-10s │ %-14s │ %-14s${NC}\n" \
    "Spheres" "Mesh Div%" "Neural Div%" "Diff" "Ratio (M/N)" "Divergent Ratio"
printf "─────────┼──────────────┼──────────────┼────────────┼────────────────┼────────────────\n"

# Print results for each sphere count
for num_spheres in "${SPHERE_COUNTS[@]}"; do
    mesh_div=${MESH_DIVERGENCE[$num_spheres]:-"N/A"}
    neural_div=${NEURAL_DIVERGENCE[$num_spheres]:-"N/A"}
    mesh_divergent=${MESH_DIVERGENT[$num_spheres]:-"N/A"}
    neural_divergent=${NEURAL_DIVERGENT[$num_spheres]:-"N/A"}

    # Calculate difference and ratios
    if [[ "$mesh_div" != "N/A" && "$neural_div" != "N/A" ]]; then
        diff_pct=$(awk "BEGIN {printf \"%.2f\", ($neural_div - $mesh_div) * 100}")

        # Divergence ratio: mesh/neural
        if awk "BEGIN {exit !($neural_div > 0)}"; then
            div_ratio=$(awk "BEGIN {printf \"%.3f\", $mesh_div / $neural_div}")
        else
            div_ratio="N/A"
        fi

        # Divergent thread count ratio: mesh/neural
        if [ "$neural_divergent" != "N/A" ] && [ "$neural_divergent" != "0" ]; then
            count_ratio=$(awk "BEGIN {printf \"%.3f\", $mesh_divergent / $neural_divergent}")
        else
            count_ratio="N/A"
        fi

        # Color code the difference
        if awk "BEGIN {exit !($diff_pct > 0)}"; then
            diff_color="${RED}"
            diff_str="+${diff_pct}%"
        elif awk "BEGIN {exit !($diff_pct < 0)}"; then
            diff_color="${GREEN}"
            diff_str="${diff_pct}%"
        else
            diff_color="${NC}"
            diff_str="0%"
        fi

        # Format ratios as percentages
        mesh_pct=$(awk "BEGIN {printf \"%.4f\", $mesh_div * 100}")
        neural_pct=$(awk "BEGIN {printf \"%.4f\", $neural_div * 100}")

        printf "%-8s │ %10s%% │ %10s%% │ ${diff_color}%10s${NC} │ %14s │ %14s\n" \
            "$num_spheres" "$mesh_pct" "$neural_pct" "$diff_str" "$div_ratio" "$count_ratio"
    else
        printf "%-8s │ %12s │ %12s │ %10s │ %14s │ %14s\n" \
            "$num_spheres" "$mesh_div" "$neural_div" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "============================================================"
echo "                    DETAILED METRICS"
echo "============================================================"
echo ""

printf "${CYAN}%-8s │ %-8s │ %-15s │ %-15s │ %-10s${NC}\n" \
    "Type" "Spheres" "Branch Targets" "Divergent" "Ratio"
printf "─────────┼──────────┼─────────────────┼─────────────────┼────────────\n"

for num_spheres in "${SPHERE_COUNTS[@]}"; do
    mesh_targets=${MESH_TARGETS[$num_spheres]:-"N/A"}
    mesh_divergent=${MESH_DIVERGENT[$num_spheres]:-"N/A"}
    mesh_div=${MESH_DIVERGENCE[$num_spheres]:-"N/A"}

    neural_targets=${NEURAL_TARGETS[$num_spheres]:-"N/A"}
    neural_divergent=${NEURAL_DIVERGENT[$num_spheres]:-"N/A"}
    neural_div=${NEURAL_DIVERGENCE[$num_spheres]:-"N/A"}

    if [[ "$mesh_div" != "N/A" ]]; then
        mesh_pct=$(awk "BEGIN {printf \"%.4f\", $mesh_div * 100}")
        printf "%-8s │ %8s │ %15s │ %15s │ %8s%%\n" \
            "mesh" "$num_spheres" "$mesh_targets" "$mesh_divergent" "$mesh_pct"
    fi

    if [[ "$neural_div" != "N/A" ]]; then
        neural_pct=$(awk "BEGIN {printf \"%.4f\", $neural_div * 100}")
        printf "%-8s │ %8s │ %15s │ %15s │ %8s%%\n" \
            "neural" "$num_spheres" "$neural_targets" "$neural_divergent" "$neural_pct"
    fi
done

# Print efficiency comparison table
echo ""
echo "============================================================"
echo "           EFFICIENCY METRICS (Branch vs Warp)"
echo "============================================================"
echo ""
echo "Branch Efficiency: % of branches where all threads took same path"
echo "Warp Efficiency:   % of threads active per instruction (avg_threads/32)"
echo "Efficiency Gap:    Branch Eff - Warp Eff (higher = longer-lived divergence)"
echo ""

printf "${CYAN}%-8s │ %-6s │ %12s │ %12s │ %12s │ %12s${NC}\n" \
    "Spheres" "Type" "Branch Eff%" "Warp Eff%" "Gap%" "Avg Threads"
printf "─────────┼────────┼──────────────┼──────────────┼──────────────┼──────────────\n"

for num_spheres in "${SPHERE_COUNTS[@]}"; do
    mesh_branch_eff=${MESH_BRANCH_EFF[$num_spheres]:-"N/A"}
    mesh_warp_eff=${MESH_WARP_EFF[$num_spheres]:-"N/A"}
    mesh_gap=${MESH_EFF_GAP[$num_spheres]:-"N/A"}
    mesh_threads=${MESH_AVG_THREADS[$num_spheres]:-"N/A"}

    neural_branch_eff=${NEURAL_BRANCH_EFF[$num_spheres]:-"N/A"}
    neural_warp_eff=${NEURAL_WARP_EFF[$num_spheres]:-"N/A"}
    neural_gap=${NEURAL_EFF_GAP[$num_spheres]:-"N/A"}
    neural_threads=${NEURAL_AVG_THREADS[$num_spheres]:-"N/A"}

    # Mesh row
    if [[ "$mesh_branch_eff" != "N/A" ]]; then
        # Color code gap - higher gap is worse (red)
        if [[ "$mesh_gap" != "N/A" ]] && awk "BEGIN {exit !($mesh_gap > 20)}"; then
            gap_color="${RED}"
        elif [[ "$mesh_gap" != "N/A" ]] && awk "BEGIN {exit !($mesh_gap > 10)}"; then
            gap_color="${YELLOW}"
        else
            gap_color="${GREEN}"
        fi
        printf "%-8s │ %-6s │ %11s%% │ %11s%% │ ${gap_color}%11s%%${NC} │ %12s\n" \
            "$num_spheres" "mesh" "$mesh_branch_eff" "$mesh_warp_eff" "$mesh_gap" "$mesh_threads"
    fi

    # Neural row
    if [[ "$neural_branch_eff" != "N/A" ]]; then
        if [[ "$neural_gap" != "N/A" ]] && awk "BEGIN {exit !($neural_gap > 20)}"; then
            gap_color="${RED}"
        elif [[ "$neural_gap" != "N/A" ]] && awk "BEGIN {exit !($neural_gap > 10)}"; then
            gap_color="${YELLOW}"
        else
            gap_color="${GREEN}"
        fi
        printf "%-8s │ %-6s │ %11s%% │ %11s%% │ ${gap_color}%11s%%${NC} │ %12s\n" \
            "" "neural" "$neural_branch_eff" "$neural_warp_eff" "$neural_gap" "$neural_threads"
    fi

    # Gap comparison row
    if [[ "$mesh_gap" != "N/A" && "$neural_gap" != "N/A" ]]; then
        gap_diff=$(awk "BEGIN {printf \"%.2f\", $mesh_gap - $neural_gap}")
        if awk "BEGIN {exit !($gap_diff > 0)}"; then
            printf "%-8s │ ${YELLOW}%-6s${NC} │ %12s │ %12s │ ${YELLOW}%+11s%%${NC} │ %12s\n" \
                "" "Δ M-N" "" "" "$gap_diff" ""
        else
            printf "%-8s │ ${YELLOW}%-6s${NC} │ %12s │ %12s │ ${GREEN}%+11s%%${NC} │ %12s\n" \
                "" "Δ M-N" "" "" "$gap_diff" ""
        fi
    fi
    printf "─────────┼────────┼──────────────┼──────────────┼──────────────┼──────────────\n"
done

# Print compute/memory metrics table
echo ""
echo "============================================================"
echo "        COMPUTE & MEMORY METRICS (Mesh vs Neural)"
echo "============================================================"
echo ""
echo "FMA Ratio:    % of instructions that are fused multiply-add (compute intensity)"
echo "L1/L2 Hit:    Cache hit rates (higher = better memory locality)"
echo "Occupancy:    % of max warps active (higher = better latency hiding)"
echo "Mem Stalls:   % of cycles stalled on memory (lower = better)"
echo ""

printf "${CYAN}%-8s │ %-6s │ %10s │ %10s │ %10s │ %10s │ %10s${NC}\n" \
    "Spheres" "Type" "FMA%" "L1 Hit%" "L2 Hit%" "Occupancy%" "Mem Stall%"
printf "─────────┼────────┼────────────┼────────────┼────────────┼────────────┼────────────\n"

for num_spheres in "${SPHERE_COUNTS[@]}"; do
    mesh_fma=${MESH_FFMA[$num_spheres]:-"N/A"}
    mesh_l1=${MESH_L1_HIT[$num_spheres]:-"N/A"}
    mesh_l2=${MESH_L2_HIT[$num_spheres]:-"N/A"}
    mesh_occ=${MESH_OCCUPANCY[$num_spheres]:-"N/A"}
    mesh_stall=${MESH_MEM_STALL[$num_spheres]:-"N/A"}

    neural_fma=${NEURAL_FFMA[$num_spheres]:-"N/A"}
    neural_l1=${NEURAL_L1_HIT[$num_spheres]:-"N/A"}
    neural_l2=${NEURAL_L2_HIT[$num_spheres]:-"N/A"}
    neural_occ=${NEURAL_OCCUPANCY[$num_spheres]:-"N/A"}
    neural_stall=${NEURAL_MEM_STALL[$num_spheres]:-"N/A"}

    # Mesh row
    if [[ "$mesh_fma" != "N/A" ]]; then
        printf "%-8s │ %-6s │ %10s │ %10s │ %10s │ %10s │ %10s\n" \
            "$num_spheres" "mesh" "$mesh_fma" "$mesh_l1" "$mesh_l2" "$mesh_occ" "$mesh_stall"
    fi

    # Neural row
    if [[ "$neural_fma" != "N/A" ]]; then
        printf "%-8s │ %-6s │ %10s │ %10s │ %10s │ %10s │ %10s\n" \
            "" "neural" "$neural_fma" "$neural_l1" "$neural_l2" "$neural_occ" "$neural_stall"
    fi

    # Delta row
    if [[ "$mesh_fma" != "N/A" && "$neural_fma" != "N/A" ]]; then
        fma_diff=$(awk "BEGIN {printf \"%.2f\", $neural_fma - $mesh_fma}")
        stall_diff=$(awk "BEGIN {printf \"%.2f\", $mesh_stall - $neural_stall}")
        # Color: positive FMA diff is good (neural more compute), positive stall diff means mesh worse
        if awk "BEGIN {exit !($fma_diff > 0)}"; then
            fma_color="${GREEN}"
        else
            fma_color="${YELLOW}"
        fi
        if awk "BEGIN {exit !($stall_diff > 0)}"; then
            stall_color="${RED}"
        else
            stall_color="${GREEN}"
        fi
        printf "%-8s │ ${YELLOW}%-6s${NC} │ ${fma_color}%+10s${NC} │ %10s │ %10s │ %10s │ ${stall_color}%+10s${NC}\n" \
            "" "Δ N-M" "$fma_diff" "" "" "" "-$stall_diff"
    fi
    printf "─────────┼────────┼────────────┼────────────┼────────────┼────────────┼────────────\n"
done

# Print registers, occupancy, and timing table
echo ""
echo "============================================================"
echo "      REGISTERS, OCCUPANCY & TIMING (Mesh vs Neural)"
echo "============================================================"
echo ""
echo "Registers/Thread: Number of registers used per thread (higher = lower occupancy)"
echo "Occupancy:        % of max warps that can be active (limited by registers/shared mem)"
echo "Duration:         Kernel execution time in milliseconds"
echo ""

printf "${CYAN}%-8s │ %-6s │ %12s │ %12s │ %12s${NC}\n" \
    "Spheres" "Type" "Registers" "Occupancy%" "Duration(ms)"
printf "─────────┼────────┼──────────────┼──────────────┼──────────────\n"

for num_spheres in "${SPHERE_COUNTS[@]}"; do
    mesh_reg=${MESH_REGISTERS[$num_spheres]:-"N/A"}
    mesh_occ=${MESH_OCCUPANCY[$num_spheres]:-"N/A"}
    mesh_dur=${MESH_DURATION[$num_spheres]:-"N/A"}

    neural_reg=${NEURAL_REGISTERS[$num_spheres]:-"N/A"}
    neural_occ=${NEURAL_OCCUPANCY[$num_spheres]:-"N/A"}
    neural_dur=${NEURAL_DURATION[$num_spheres]:-"N/A"}

    # Mesh row
    if [[ "$mesh_reg" != "N/A" ]]; then
        printf "%-8s │ %-6s │ %12s │ %11s%% │ %12s\n" \
            "$num_spheres" "mesh" "$mesh_reg" "$mesh_occ" "$mesh_dur"
    fi

    # Neural row
    if [[ "$neural_reg" != "N/A" ]]; then
        printf "%-8s │ %-6s │ %12s │ %11s%% │ %12s\n" \
            "" "neural" "$neural_reg" "$neural_occ" "$neural_dur"
    fi

    # Delta row with speedup calculation
    if [[ "$mesh_dur" != "N/A" && "$neural_dur" != "N/A" && "$mesh_dur" != "0" && "$neural_dur" != "0" ]]; then
        reg_diff=$(awk "BEGIN {printf \"%+d\", $neural_reg - $mesh_reg}")
        occ_diff=$(awk "BEGIN {printf \"%+.1f\", $neural_occ - $mesh_occ}")
        # Calculate speedup: mesh_dur / neural_dur (>1 means neural is faster)
        speedup=$(awk "BEGIN {printf \"%.2f\", $mesh_dur / $neural_dur}")

        # Color: more registers is bad (red), higher occupancy is good (green)
        if awk "BEGIN {exit !($neural_reg > $mesh_reg)}"; then
            reg_color="${RED}"
        else
            reg_color="${GREEN}"
        fi
        if awk "BEGIN {exit !($neural_occ > $mesh_occ)}"; then
            occ_color="${GREEN}"
        else
            occ_color="${RED}"
        fi
        # Speedup > 1 means neural is faster (green), < 1 means mesh is faster (red)
        if awk "BEGIN {exit !($speedup > 1)}"; then
            speed_color="${GREEN}"
            speed_str="${speedup}x faster"
        else
            speed_color="${RED}"
            inv_speedup=$(awk "BEGIN {printf \"%.2f\", 1 / $speedup}")
            speed_str="${inv_speedup}x slower"
        fi

        printf "%-8s │ ${YELLOW}%-6s${NC} │ ${reg_color}%12s${NC} │ ${occ_color}%12s${NC} │ ${speed_color}%12s${NC}\n" \
            "" "Δ N-M" "$reg_diff" "$occ_diff" "$speed_str"
    fi
    printf "─────────┼────────┼──────────────┼──────────────┼──────────────\n"
done

echo ""
echo "============================================================"
echo "Results saved to: $RESULTS_FILE"
echo "============================================================"
