#!/usr/bin/env python3
"""
Parse NCU cuda,sass output with source line correlation.

Usage:
    # Export from ncu with -G debug flag enabled during kernel compilation:
    ncu --import output/render.ncu-rep --page source --print-source cuda,sass > /tmp/sass.txt

    # Then run this script:
    python3 scripts/parse_ncu_cuda_sass.py /tmp/sass.txt [source_file_filter]
"""

import sys
import re
from collections import defaultdict
from pathlib import Path


def parse_cuda_sass_output(content):
    """
    Parse ncu --print-source cuda,sass output.

    Format:
    - "File Path: /path/to/file.h" headers before each file section
    - "Function Name: funcName" headers before each function section
    - Line number + source code in first column
    - Metrics in subsequent columns
    """
    results = []
    current_file = None
    current_function = None
    lines = content.split('\n')

    # Find column indices from header line
    # Look for header pattern with metric names
    header_pattern = re.compile(r'Address\s+Source.*Diverg')

    for i, line in enumerate(lines):
        # Check for file path header
        if line.startswith('File Path:'):
            current_file = line.split(':', 1)[1].strip()
            continue

        # Check for function name header (format: "Function Name: xxx" inside /* */ comments)
        if 'Function Name:' in line:
            match = re.search(r'Function Name:\s*(.+?)(?:\s*\*/|\s*$)', line)
            if match:
                current_function = match.group(1).strip()
            continue

        # Skip header lines and separators
        if line.startswith('Address') or line.startswith('-') or not line.strip():
            continue
        if 'tall S' in line or 'amplin' in line or 'Sampl' in line:
            continue

        # Look for source lines: start with line number
        match = re.match(r'^(\d+)\s+(.+?)(?:\s{2,}|\s+(?=\d))', line)
        if match:
            line_num = int(match.group(1))
            source_text = match.group(2).strip()

            # Extract numeric columns (metrics)
            # Find where the numbers start after the source text
            rest = line[match.end():]
            parts = rest.split()

            # Parse metrics - handle truncated columns
            try:
                metrics = []
                for p in parts:
                    try:
                        metrics.append(int(p))
                    except ValueError:
                        if p == '-':
                            metrics.append(0)
                        # Skip non-numeric (like "Local", "Store", etc)

                if len(metrics) >= 10:
                    # Column mapping (based on header):
                    # 0: Warp Stalls (All Samples)
                    # 1: Warp Stalls (Not-issued)
                    # 2: # Samples
                    # 3: Instructions Executed
                    # 4: Thread Instructions Executed
                    # 5: Predicated-On Thread Instructions Executed
                    # 6: Avg. Threads Executed
                    # 7: Avg. Predicated-On Threads Executed
                    # 8: Divergent Branches

                    warp_stalls = metrics[0]
                    samples = metrics[2] if len(metrics) > 2 else 0
                    instructions = metrics[3] if len(metrics) > 3 else 0
                    thread_instructions = metrics[4] if len(metrics) > 4 else 0
                    avg_threads = metrics[6] if len(metrics) > 6 else 32
                    divergent = metrics[8] if len(metrics) > 8 else 0

                    results.append({
                        'file': current_file,
                        'function': current_function,
                        'line': line_num,
                        'source': source_text[:80],
                        'warp_stalls': warp_stalls,
                        'samples': samples,
                        'instructions': instructions,
                        'thread_instructions': thread_instructions,
                        'avg_threads': avg_threads,
                        'divergent': divergent,
                    })
            except (ValueError, IndexError):
                continue

    return results


def aggregate_by_source(results, file_filter=None):
    """Aggregate metrics by source file, function, and line."""
    aggregated = defaultdict(lambda: {
        'source': '',
        'function': '',
        'warp_stalls': 0,
        'samples': 0,
        'instructions': 0,
        'thread_instructions': 0,
        'divergent': 0,
        'count': 0,
    })

    for r in results:
        if file_filter and file_filter not in (r['file'] or ''):
            continue

        key = (r['file'], r['line'])
        agg = aggregated[key]
        agg['source'] = r['source']
        agg['function'] = r.get('function', '')
        agg['warp_stalls'] += r['warp_stalls']
        agg['samples'] += r['samples']
        agg['instructions'] += r.get('instructions', 0)
        agg['thread_instructions'] += r.get('thread_instructions', 0)
        agg['divergent'] += r['divergent']
        agg['count'] += 1

    return aggregated


def aggregate_by_function(results, file_filter=None):
    """Aggregate metrics by function name."""
    aggregated = defaultdict(lambda: {
        'file': '',
        'warp_stalls': 0,
        'samples': 0,
        'instructions': 0,
        'thread_instructions': 0,
        'divergent': 0,
        'count': 0,
    })

    for r in results:
        if file_filter and file_filter not in (r['file'] or ''):
            continue

        func = r.get('function', 'unknown')
        if func:
            agg = aggregated[func]
            agg['file'] = r['file'] or ''
            agg['warp_stalls'] += r['warp_stalls']
            agg['samples'] += r['samples']
            agg['instructions'] += r.get('instructions', 0)
            agg['thread_instructions'] += r.get('thread_instructions', 0)
            agg['divergent'] += r['divergent']
            agg['count'] += 1

    return aggregated


def calc_warp_efficiency(thread_instructions, instructions):
    """
    Calculate warp efficiency as percentage.

    Warp efficiency = (thread_instructions / (instructions * 32)) * 100

    This tells us what percentage of threads were active on average.
    32 threads per warp is the maximum.
    """
    if instructions == 0:
        return 0.0
    return (thread_instructions / (instructions * 32)) * 100.0


def calc_divergence_pct(divergent, samples):
    """
    DEPRECATED: This calculation is misleading.

    The 'divergent' column in SASS output is actually a sum of per-instruction
    averages, not a count of divergent branches. Dividing by samples gives
    meaningless results like 227%.

    Use calc_warp_efficiency() instead for meaningful divergence analysis.
    """
    if samples == 0:
        return 0.0
    return (divergent / samples) * 100.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_ncu_cuda_sass.py <cuda_sass_output.txt> [source_file_filter]")
        print("")
        print("Example:")
        print("  ncu --import output/render.ncu-rep --page source --print-source cuda,sass > /tmp/sass.txt")
        print("  python3 scripts/parse_ncu_cuda_sass.py /tmp/sass.txt render_kernel")
        sys.exit(1)

    input_file = sys.argv[1]
    file_filter = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Parsing {input_file}...")
    with open(input_file, 'r') as f:
        content = f.read()

    results = parse_cuda_sass_output(content)
    print(f"Found {len(results)} source line entries")

    # Get unique files
    files = set(r['file'] for r in results if r['file'])
    print(f"\nSource files found ({len(files)}):")
    for f in sorted(files):
        print(f"  {f}")

    # Get unique functions
    functions = set(r.get('function', '') for r in results if r.get('function'))
    print(f"\nFunctions found ({len(functions)}):")
    for f in sorted(functions)[:20]:  # Limit to 20
        print(f"  {f}")
    if len(functions) > 20:
        print(f"  ... and {len(functions) - 20} more")

    # Aggregate by source
    aggregated = aggregate_by_source(results, file_filter)

    # Aggregate by function for summary
    func_aggregated = aggregate_by_function(results, file_filter)

    if not aggregated:
        print(f"\nNo results found" + (f" matching '{file_filter}'" if file_filter else ""))
        return

    # Sort by divergent branches (primary), then warp stalls
    sorted_by_divergent = sorted(
        aggregated.items(),
        key=lambda x: (x[1]['divergent'], x[1]['warp_stalls']),
        reverse=True
    )

    # Sort by warp stalls for comparison
    sorted_by_stalls = sorted(
        aggregated.items(),
        key=lambda x: x[1]['warp_stalls'],
        reverse=True
    )

    # Function hotspots
    print(f"\n{'='*80}")
    print(f"Function Hotspots by WARP EFFICIENCY" + (f" (filter: {file_filter})" if file_filter else ""))
    print(f"{'='*80}")

    sorted_funcs = sorted(
        func_aggregated.items(),
        key=lambda x: (x[1]['thread_instructions'], x[1]['warp_stalls']),
        reverse=True
    )

    print(f"\n{'Instructions':<12} {'Thread Inst':<14} {'Warp Eff%':<10} {'Stalls':<10} Function")
    print("-" * 100)
    shown = 0
    for func_name, metrics in sorted_funcs:
        if metrics['instructions'] > 0 or metrics['warp_stalls'] > 100:
            warp_eff = calc_warp_efficiency(metrics['thread_instructions'], metrics['instructions'])
            print(f"{metrics['instructions']:<12} {metrics['thread_instructions']:<14} {warp_eff:<10.2f} {metrics['warp_stalls']:<10} {func_name}")
            shown += 1
            if shown >= 20:
                break

    # Source line hotspots - sorted by lowest warp efficiency (most divergent)
    print(f"\n{'='*80}")
    print(f"Source Line Hotspots by LOWEST WARP EFFICIENCY" + (f" (filter: {file_filter})" if file_filter else ""))
    print(f"{'='*80}")

    # Sort by warp efficiency (ascending) - lowest efficiency first
    sorted_by_efficiency = sorted(
        [(k, v) for k, v in aggregated.items() if v['instructions'] > 100],
        key=lambda x: calc_warp_efficiency(x[1]['thread_instructions'], x[1]['instructions']),
        reverse=False
    )

    print(f"\n{'File:Line':<35} {'Warp Eff%':<10} {'Instructions':<12} {'Thread Inst':<14}")
    print("-" * 80)
    shown = 0
    for (file_path, line_num), metrics in sorted_by_efficiency:
        warp_eff = calc_warp_efficiency(metrics['thread_instructions'], metrics['instructions'])
        if warp_eff < 95:  # Only show lines with notable divergence
            short_file = Path(file_path).name if file_path else "?"
            loc = f"{short_file}:{line_num}"
            func = metrics.get('function', '')
            print(f"{loc:<35} {warp_eff:<10.2f} {metrics['instructions']:<12} {metrics['thread_instructions']:<14}")
            print(f"    Function: {func}")
            print(f"    Source:   {metrics['source'][:80]}")
            shown += 1
            if shown >= 30:
                break

    print(f"\n{'='*80}")
    print(f"Source Line Hotspots by WARP STALLS")
    print(f"{'='*80}")

    print(f"\n{'File:Line':<35} {'Stalls':<10} {'Warp Eff%':<10} {'Instructions':<12}")
    print("-" * 80)
    shown = 0
    for (file_path, line_num), metrics in sorted_by_stalls:
        if metrics['warp_stalls'] > 100:
            short_file = Path(file_path).name if file_path else "?"
            loc = f"{short_file}:{line_num}"
            func = metrics.get('function', '')
            warp_eff = calc_warp_efficiency(metrics['thread_instructions'], metrics['instructions'])
            print(f"{loc:<35} {metrics['warp_stalls']:<10} {warp_eff:<10.2f} {metrics['instructions']:<12}")
            print(f"    Function: {func}")
            print(f"    Source:   {metrics['source'][:80]}")
            shown += 1
            if shown >= 20:
                break

    # Summary stats
    total_stalls = sum(m['warp_stalls'] for m in aggregated.values())
    total_instructions = sum(m['instructions'] for m in aggregated.values())
    total_thread_instructions = sum(m['thread_instructions'] for m in aggregated.values())
    overall_warp_eff = calc_warp_efficiency(total_thread_instructions, total_instructions)

    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"Total instructions:         {total_instructions:,}")
    print(f"Total thread instructions:  {total_thread_instructions:,}")
    print(f"Overall warp efficiency:    {overall_warp_eff:.2f}%")
    print(f"  (100% = all 32 threads active, lower = more divergence)")
    print(f"Total warp stalls:          {total_stalls:,}")
    print(f"Unique functions:           {len(func_aggregated)}")
    print(f"Unique source lines:        {len(aggregated)}")


if __name__ == '__main__':
    main()
