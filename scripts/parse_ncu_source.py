#!/usr/bin/env python3
"""
Parse NCU SASS output and correlate with PTX line info to map back to source lines.

Usage:
    # First export SASS from ncu:
    ncu --import output/render_only.ncu-rep --page source --print-source sass > /tmp/sass.txt

    # Then run this script:
    python3 scripts/parse_ncu_source.py cache/*.bin render_kernel /tmp/sass.txt
"""

import sys
import re
from collections import defaultdict
from pathlib import Path


def parse_ptx_file_table(ptx_content):
    """Extract .file directives mapping file IDs to paths."""
    file_table = {}
    for match in re.finditer(r'\.file\s+(\d+)\s+"([^"]+)"', ptx_content):
        file_id = int(match.group(1))
        file_path = match.group(2)
        file_table[file_id] = file_path
    return file_table


def parse_ptx_loc_directives(ptx_content):
    """
    Parse PTX to extract .loc directives with their PTX line positions.
    Returns list of (ptx_line_num, file_id, source_line, column)
    """
    locations = []
    lines = ptx_content.split('\n')

    for i, line in enumerate(lines):
        loc_match = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if loc_match:
            file_id = int(loc_match.group(1))
            source_line = int(loc_match.group(2))
            column = int(loc_match.group(3))
            locations.append((i, file_id, source_line, column))

    return locations


def parse_sass_output(sass_content):
    """
    Parse ncu SASS output to extract address, instruction, and key metrics.

    Format (columns are whitespace-separated):
    Address  Instruction  WarpStalls(All)  WarpStalls(NotIssued)  #Samples  ...  DivergentBranches  ...
    """
    instructions = []
    lines = sass_content.strip().split('\n')

    for line in lines:
        # Look for lines starting with 0x (addresses)
        stripped = line.strip()
        if not stripped.startswith('0x'):
            continue

        # Split by multiple spaces to handle variable column widths
        # The format is: Address  Instruction  Metrics...
        parts = stripped.split()
        if len(parts) < 10:
            continue

        try:
            addr = int(parts[0], 16)

            # Find instruction text (everything after address until first numeric column)
            instr_parts = []
            metric_start_idx = 1
            for idx, p in enumerate(parts[1:], start=1):
                # Check if this looks like a number (metric)
                try:
                    int(p)
                    metric_start_idx = idx
                    break
                except ValueError:
                    instr_parts.append(p)

            instruction = ' '.join(instr_parts)

            # Extract numeric metrics
            metrics = []
            for p in parts[metric_start_idx:]:
                try:
                    metrics.append(int(p))
                except ValueError:
                    # Non-numeric (like "-" or "Local")
                    metrics.append(0)

            # Key metrics based on ncu output format:
            # [0] = Warp Stall Samples (All)
            # [1] = Warp Stall Samples (Not-issued)
            # [2] = # Samples
            # [3] = Instructions Executed
            # [4] = Thread Instructions Executed
            # [5] = Predicated-On Thread Instructions Executed
            # [6] = Avg. Threads Executed
            # [7] = Avg. Predicated-On Threads Executed
            # [8] = Divergent Branches

            warp_stalls = metrics[0] if len(metrics) > 0 else 0
            samples = metrics[2] if len(metrics) > 2 else warp_stalls
            divergent = metrics[8] if len(metrics) > 8 else 0

            instructions.append({
                'address': addr,
                'instruction': instruction,
                'warp_stalls': warp_stalls,
                'samples': samples,
                'divergent': divergent,
            })
        except (ValueError, IndexError):
            continue

    return instructions


def build_address_to_source_map(ptx_content):
    """
    Build a mapping from SASS instruction index to source line.

    Strategy:
    1. Parse PTX to get instruction count per source line
    2. SASS instructions are roughly sequential with PTX
    3. Use the .loc directives to establish source line boundaries
    """
    # Get all .loc directives
    loc_pattern = re.compile(r'\.loc\s+(\d+)\s+(\d+)\s+(\d+)')

    # Build a list of (ptx_instruction_index, file_id, source_line)
    # where ptx_instruction_index is the count of non-.loc, non-directive lines

    ptx_lines = ptx_content.split('\n')

    loc_transitions = []  # (ptx_instr_count, file_id, source_line)
    current_loc = None
    ptx_instr_count = 0

    for line in ptx_lines:
        stripped = line.strip()

        # Check for .loc directive
        loc_match = loc_pattern.match(stripped)
        if loc_match:
            current_loc = (int(loc_match.group(1)), int(loc_match.group(2)))
            continue

        # Skip empty lines, comments, and directives
        if not stripped or stripped.startswith('//') or stripped.startswith('.'):
            continue

        # This is a PTX instruction
        if current_loc:
            loc_transitions.append((ptx_instr_count, current_loc[0], current_loc[1]))
        ptx_instr_count += 1

    return loc_transitions, ptx_instr_count


def correlate_to_source(sass_instructions, loc_transitions, total_ptx_instrs, target_file_id):
    """
    Map SASS instructions to source lines using PTX .loc info.

    Returns dict: source_line -> {stalls, samples, divergent, instruction_count}
    """
    source_metrics = defaultdict(lambda: {
        'warp_stalls': 0,
        'samples': 0,
        'divergent': 0,
        'count': 0,
        'instructions': []
    })

    if not sass_instructions or not loc_transitions:
        return source_metrics

    num_sass = len(sass_instructions)
    num_ptx = total_ptx_instrs

    if num_ptx == 0:
        return source_metrics

    # Build a mapping: for each PTX instruction range, what source line is it?
    # loc_transitions is sorted by PTX instruction index

    # Create array mapping ptx_instr_idx -> (file_id, source_line)
    ptx_to_source = [None] * num_ptx

    for i, (ptx_idx, file_id, source_line) in enumerate(loc_transitions):
        # Find end of this source line's range
        if i + 1 < len(loc_transitions):
            end_idx = loc_transitions[i + 1][0]
        else:
            end_idx = num_ptx

        # Mark all PTX instructions in this range
        for j in range(ptx_idx, min(end_idx, num_ptx)):
            ptx_to_source[j] = (file_id, source_line)

    # Now map SASS to source using linear scaling
    # Assumption: SASS instructions roughly correspond to PTX instructions sequentially
    ratio = num_ptx / num_sass if num_sass > 0 else 1

    for sass_idx, sass_instr in enumerate(sass_instructions):
        # Estimate corresponding PTX instruction
        ptx_idx = min(int(sass_idx * ratio), num_ptx - 1)

        if ptx_to_source[ptx_idx]:
            file_id, source_line = ptx_to_source[ptx_idx]

            if file_id == target_file_id:
                source_metrics[source_line]['warp_stalls'] += sass_instr['warp_stalls']
                source_metrics[source_line]['samples'] += sass_instr['samples']
                source_metrics[source_line]['divergent'] += sass_instr['divergent']
                source_metrics[source_line]['count'] += 1
                if sass_instr['warp_stalls'] > 50:  # Only track high-stall instructions
                    source_metrics[source_line]['instructions'].append(
                        f"{sass_instr['instruction'][:30]} (stalls={sass_instr['warp_stalls']})"
                    )

    return source_metrics


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 parse_ncu_source.py <ptx_file> <source_file_name> [sass_output.txt]")
        print("")
        print("Example (PTX analysis only):")
        print("  python3 scripts/parse_ncu_source.py cache/*.bin render_kernel")
        print("")
        print("Example (with SASS metrics from ncu):")
        print("  ncu --import output/render_only.ncu-rep --page source --print-source sass > /tmp/sass.txt")
        print("  python3 scripts/parse_ncu_source.py cache/*.bin render_kernel /tmp/sass.txt")
        sys.exit(1)

    ptx_file = sys.argv[1]
    source_name = sys.argv[2]
    sass_file = sys.argv[3] if len(sys.argv) > 3 else None

    # Read PTX
    print(f"Parsing PTX from {ptx_file}...")
    with open(ptx_file, 'r', errors='ignore') as f:
        ptx_content = f.read()

    file_table = parse_ptx_file_table(ptx_content)
    print(f"  Found {len(file_table)} source files:")
    for fid, fpath in sorted(file_table.items()):
        print(f"    {fid}: {fpath}")

    # Find target file ID
    target_file_id = None
    source_path = None
    for fid, fpath in file_table.items():
        if source_name in fpath:
            target_file_id = fid
            source_path = fpath
            break

    if target_file_id is None:
        print(f"\nWarning: Could not find '{source_name}' in file table, defaulting to file 1")
        target_file_id = 1
        source_path = file_table.get(1)

    # Build PTX instruction to source mapping
    loc_transitions, total_ptx_instrs = build_address_to_source_map(ptx_content)
    print(f"  Found {len(loc_transitions)} source location transitions")
    print(f"  Total PTX instructions: {total_ptx_instrs}")

    # Parse SASS if provided
    sass_instructions = []
    if sass_file:
        print(f"\nParsing SASS from {sass_file}...")
        with open(sass_file, 'r') as f:
            sass_content = f.read()
        sass_instructions = parse_sass_output(sass_content)
        print(f"  Found {len(sass_instructions)} SASS instructions")

        if sass_instructions:
            total_stalls = sum(i['warp_stalls'] for i in sass_instructions)
            total_divergent = sum(i['divergent'] for i in sass_instructions)
            print(f"  Total warp stalls: {total_stalls}")
            print(f"  Total divergent branches: {total_divergent}")

    # Correlate SASS to source
    if sass_instructions:
        print(f"\n{'='*80}")
        print(f"Source line hotspots by RUNTIME METRICS (file: {source_name})")
        print(f"{'='*80}")

        source_metrics = correlate_to_source(
            sass_instructions, loc_transitions, total_ptx_instrs, target_file_id
        )

        # Sort by divergent branches (primary), then by warp stalls (secondary)
        sorted_by_divergent = sorted(
            source_metrics.items(),
            key=lambda x: (x[1]['divergent'], x[1]['warp_stalls']),
            reverse=True
        )

        # Also sort by stalls for comparison
        sorted_by_stalls = sorted(
            source_metrics.items(),
            key=lambda x: x[1]['warp_stalls'],
            reverse=True
        )

        print(f"\n=== SORTED BY DIVERGENT BRANCHES ===")
        print(f"{'Line':<8} {'Divergent':<10} {'Stalls':<10} {'Samples':<10} {'#Instrs':<8}")
        print("-" * 50)
        shown = 0
        for line_num, metrics in sorted_by_divergent:
            if metrics['divergent'] > 0 or metrics['warp_stalls'] > 0:
                print(f"{line_num:<8} {metrics['divergent']:<10} {metrics['warp_stalls']:<10} {metrics['samples']:<10} {metrics['count']:<8}")
                shown += 1
                if shown >= 30:
                    break

        print(f"\n=== SORTED BY WARP STALLS ===")
        print(f"{'Line':<8} {'Stalls':<10} {'Divergent':<10} {'Samples':<10} {'#Instrs':<8}")
        print("-" * 50)
        shown = 0
        for line_num, metrics in sorted_by_stalls:
            if metrics['warp_stalls'] > 0:
                print(f"{line_num:<8} {metrics['warp_stalls']:<10} {metrics['divergent']:<10} {metrics['samples']:<10} {metrics['count']:<8}")
                shown += 1
                if shown >= 20:
                    break

        # Show actual source for top divergent hotspots
        if source_path and Path(source_path).exists():
            print(f"\n{'='*80}")
            print(f"Top DIVERGENT source lines (sorted by divergent branch count):")
            print(f"{'='*80}\n")

            with open(source_path, 'r') as f:
                source_lines = f.readlines()

            shown = 0
            context_lines = 3  # Show N lines before and after
            for line_num, metrics in sorted_by_divergent:
                if (metrics['divergent'] > 0 or metrics['warp_stalls'] > 0) and 0 < line_num <= len(source_lines):
                    print(f"--- Line {line_num} (divergent={metrics['divergent']}, stalls={metrics['warp_stalls']}) ---")

                    # Show context: lines before and after
                    start = max(0, line_num - context_lines - 1)
                    end = min(len(source_lines), line_num + context_lines)

                    for i in range(start, end):
                        line_indicator = ">>>" if i == line_num - 1 else "   "
                        src = source_lines[i].rstrip()
                        print(f"  {line_indicator} {i+1:4d}: {src[:100]}")

                    if metrics['instructions']:
                        print(f"  Hot SASS: {metrics['instructions'][0]}")
                    print()
                    shown += 1
                    if shown >= 12:
                        break

    else:
        # Fallback: just show PTX instruction count per line
        print(f"\n{'='*80}")
        print(f"Source line complexity by PTX instruction count (no SASS metrics)")
        print(f"{'='*80}")

        line_counts = defaultdict(int)
        for _, file_id, source_line in loc_transitions:
            if file_id == target_file_id:
                line_counts[source_line] += 1

        sorted_lines = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'Line':<8} {'PTX Instrs':<12}")
        print("-" * 25)
        for line_num, count in sorted_lines[:30]:
            print(f"{line_num:<8} {count:<12}")

        # Show actual source
        if source_path and Path(source_path).exists():
            print(f"\n{'='*80}")
            print(f"Top complex source lines:")
            print(f"{'='*80}\n")

            with open(source_path, 'r') as f:
                source_lines = f.readlines()

            for line_num, count in sorted_lines[:10]:
                if 0 < line_num <= len(source_lines):
                    src = source_lines[line_num - 1].rstrip()
                    print(f"Line {line_num} ({count} PTX instructions):")
                    print(f"  {src[:100]}")
                    print()


if __name__ == '__main__':
    main()
