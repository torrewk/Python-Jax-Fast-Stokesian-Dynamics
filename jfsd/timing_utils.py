"""Class and utilities for timing the simulation."""
import time
from typing import Dict, NamedTuple, Optional
from collections import defaultdict
import numpy as np
from loguru import logger

class TimerStats(NamedTuple):
    """Statistics for a timer section."""
    total_time: float = 0.0
    mean_time: float = 0.0  
    std_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    num_calls: int = 0
    percent_total: float = 0.0
    percent_parent: float = 0.0
    parent: str = ""
    depth: int = 0

class SimulationTimer:
    """Timing utility for the simulation."""
    
    def __init__(self):
        self._timings = defaultdict(list)
        self._start_times = {}
        self._nested_calls = defaultdict(int)
        self._parents = {}  # Maps section to its parent
        self._current_stack = []  # Stack of active sections
    
    def start(self, section: str):
        """Start timing a section."""
        # Record parent-child relationship
        if self._current_stack:
            self._parents[section] = self._current_stack[-1]
        
        self._nested_calls[section] += 1
        if self._nested_calls[section] == 1:
            self._start_times[section] = time.time()
            self._current_stack.append(section)
        
    def stop(self, section: str):
        """Stop timing a section and record the duration."""
        if section not in self._nested_calls or self._nested_calls[section] <= 0:
            logger.warning(f"Stopping timer '{section}' that was never started")
            return
            
        self._nested_calls[section] -= 1
        if self._nested_calls[section] == 0:
            if section in self._start_times:
                duration = time.time() - self._start_times[section]
                self._timings[section].append(duration)
                del self._start_times[section]
                
                # Remove from current stack
                if self._current_stack and self._current_stack[-1] == section:
                    self._current_stack.pop()
    
    def get_statistics(self) -> Dict[str, TimerStats]:
        """Calculate timing statistics for each section."""
        stats = {}
        
        # First pass: Get total times per section
        section_totals = {}
        for section, times in self._timings.items():
            section_totals[section] = sum(times)
        
        # Find root sections (those without parents)
        root_sections = set(self._timings.keys()) - set(self._parents.keys())
        total_root_time = sum(section_totals[section] for section in root_sections) if root_sections else 1.0
        
        # Build children map and calculate depths
        children = defaultdict(list)
        depths = {}
        
        def calculate_depth(section, current_depth=0):
            depths[section] = current_depth
            for child in children.get(section, []):
                calculate_depth(child, current_depth + 1)
        
        for child, parent in self._parents.items():
            children[parent].append(child)
        
        # Calculate depths
        for root in root_sections:
            calculate_depth(root)
        
        # Second pass: Calculate statistics
        for section, times in self._timings.items():
            section_total = section_totals[section]
            section_mean = np.mean(times)
            section_std = np.std(times) if len(times) > 1 else 0
            section_min = min(times) if times else 0
            section_max = max(times) if times else 0
            
            # Calculate percentage of total simulation time
            section_percent = (section_total / total_root_time * 100) if total_root_time > 0 else 0
            
            # Calculate percentage of parent time
            parent = self._parents.get(section, "")
            parent_percent = 0.0
            if parent and parent in section_totals and section_totals[parent] > 0:
                parent_percent = section_total / section_totals[parent] * 100
            
            stats[section] = TimerStats(
                total_time=section_total,
                mean_time=section_mean,
                std_time=section_std,
                min_time=section_min,
                max_time=section_max,
                num_calls=len(times),
                percent_total=section_percent,
                percent_parent=parent_percent,
                parent=parent,
                depth=depths.get(section, 0)
            )
        
        return stats

    def print_report(self, top_n: Optional[int] = None, sort_by: str = "total_time"):
        """
        Print a formatted performance report.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top sections to display. If None, all sections are shown.
        sort_by : str
            How to sort results: "total_time", "mean_time", "calls", or "percent"
        """
        # By default, use the hierarchical report which is clearer
        self._print_hierarchical_report(top_n, sort_by)
        
    def print_flat_report(self, top_n: Optional[int] = None, sort_by: str = "total_time"):
        """
        Print a flat (non-hierarchical) performance report.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top sections to display. If None, all sections are shown.
        sort_by : str
            How to sort results: "total_time", "mean_time", "calls", or "percent"
        """
        stats = self.get_statistics()
        
        # Sort sections based on the requested criterion
        if sort_by == "total_time":
            sections = sorted(stats.items(), key=lambda x: x[1].total_time, reverse=True)
        elif sort_by == "mean_time":
            sections = sorted(stats.items(), key=lambda x: x[1].mean_time, reverse=True)
        elif sort_by == "calls":
            sections = sorted(stats.items(), key=lambda x: x[1].num_calls, reverse=True)
        elif sort_by == "percent":
            sections = sorted(stats.items(), key=lambda x: x[1].percent_total, reverse=True)
        else:
            sections = sorted(stats.items(), key=lambda x: x[1].total_time, reverse=True)
            
        if top_n:
            sections = sections[:top_n]
        
        # Find root sections (those without parents)
        root_sections = {section for section, stat in stats.items() if not stat.parent}
        total_root_time = sum(stats[section].total_time for section in root_sections)
        
        logger.info("\nFlat Performance Report")
        logger.info("=" * 120)
        logger.info(f"{'Section':<40} {'Total(s)':<10} {'Mean(ms)':<10} {'Calls':<8} {'Nested':<8} {'% of Parent':<14}")
        logger.info("-" * 120)
        
        for section, timing in sections:
            # Indent section names based on depth
            indent = "  " * timing.depth
            section_display = f"{indent}{section}"
            
            # Show whether this timer has child timers
            has_children = any(stat.parent == section for _, stat in stats.items())
            nested_info = "contains" if has_children else "-"
            
            # Display parent percentage
            parent_str = f"{timing.percent_parent:>7.1f}% of {timing.parent}" if timing.parent else "-"
            
            logger.info(f"{section_display:<40} {timing.total_time:>10.3f} {timing.mean_time*1000:>10.1f} "
                  f"{timing.num_calls:>8d} {nested_info:>8} {parent_str:<14}")
                  
        logger.info("-" * 120)
        logger.info(f"{'Total':<40} {total_root_time:>10.3f}")
        logger.info("=" * 120)

    def _print_hierarchical_report(self, top_n: Optional[int] = None, sort_by: str = "total_time"):
        """
        Print a hierarchical tree view of the timers.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top root sections to display. If None, all are shown.
        sort_by : str
            How to sort sections: "total_time", "mean_time", "calls", or "percent"
        """
        stats = self.get_statistics()
        
        # Build children map
        children = defaultdict(list)
        for section, timing in stats.items():
            if timing.parent:
                children[timing.parent].append(section)
        
        # Find root sections
        roots = [section for section, timing in stats.items() if not timing.parent]
        
        # Sort roots based on criterion
        if sort_by == "total_time":
            roots = sorted(roots, key=lambda s: stats[s].total_time, reverse=True)
        elif sort_by == "mean_time":
            roots = sorted(roots, key=lambda s: stats[s].mean_time, reverse=True)
        elif sort_by == "calls":
            roots = sorted(roots, key=lambda s: stats[s].num_calls, reverse=True)
        elif sort_by == "percent":
            roots = sorted(roots, key=lambda s: stats[s].percent_total, reverse=True)
        
        # Limit to top_n root sections if specified
        if top_n:
            roots = roots[:top_n]
        
        # Total time from root sections
        total_time = sum(stats[root].total_time for root in roots)
        
        logger.info("\nHierarchical Performance Report")
        logger.info("=" * 100)
        logger.info(f"{'Section':<50} {'Total(s)':<10} {'%Root':<8} {'Calls':<8} {'Mean(ms)':<10}")
        logger.info("-" * 100)
        
        # Track which sections are the last child of their parent for tree drawing
        is_last_child = {}
        
        def mark_last_children():
            """Mark which children are the last ones of their parent"""
            for parent, child_list in children.items():
                if child_list:  # Make sure the list is not empty
                    sorted_children = sorted(child_list, key=lambda s: stats[s].total_time, reverse=True)
                    for i, child in enumerate(sorted_children):
                        is_last_child[child] = (i == len(sorted_children) - 1)
        
        # Pre-compute which sections are last children
        mark_last_children()
        
        def print_section(section, depth=0):
            timing = stats[section]
            
            # Create the indent with tree symbols
            if depth == 0:
                indent = ""
                box = ""
            else:
                indent = "  " * (depth - 1)
                # Use the pre-computed is_last_child dictionary to determine the box character
                box = "└─ " if is_last_child.get(section, False) else "├─ "
            
            section_name = f"{indent}{box}{section}"
            
            # Calculate percentage of root time
            if timing.parent:
                root_finder = timing.parent
                while stats.get(root_finder, TimerStats(0, 0, 0, 0, 0, 0, 0)).parent:  # Find the root ancestor
                    root_finder = stats[root_finder].parent
                root_time = stats[root_finder].total_time
                root_percent = timing.total_time / root_time * 100 if root_time > 0 else 0
            else:
                # This is a root section
                root_percent = timing.percent_total
                
            logger.info(f"{section_name:<50} {timing.total_time:>10.3f} {root_percent:>7.1f}% "
                  f"{timing.num_calls:>8d} {timing.mean_time*1000:>10.1f}")
            
            # Sort and print children
            child_list = sorted(children[section], key=lambda s: stats[s].total_time, reverse=True)
            for child in child_list:
                print_section(child, depth + 1)
        
        # Print each root section and its children
        for root in roots:
            print_section(root)
            
        logger.info("-" * 100)
        logger.info(f"{'Total':<50} {total_time:>10.3f}")
        logger.info("=" * 100)
        logger.info("Note: '%Root' shows percentage relative to the root timer's time in each hierarchy.")
        
        # Add explanation when multiple root timers exist
        if len(roots) > 1:
            logger.info(f"Note: There are {len(roots)} separate top-level timers. Each top-level timer's")
            logger.info(f"      percentage is relative to the total execution time ({total_time:.3f}s).")

    def get_section_time(self, section: str) -> float:
        """Get the total time spent in a specific section."""
        times = self._timings.get(section, [])
        return sum(times)
        
    def reset(self):
        """Reset all timing data."""
        self._timings = defaultdict(list)
        self._start_times = {}
        self._nested_calls = defaultdict(int)
        self._parents = {}
        self._current_stack = []

    def print_summary(self):
        """Print a simple summary of top-level timers."""
        stats = self.get_statistics()
        
        # Find root sections
        root_sections = [s for s, t in stats.items() if not t.parent]
        root_sections = sorted(root_sections, key=lambda s: stats[s].total_time, reverse=True)
        
        total_time = sum(stats[root].total_time for root in root_sections)
        
        logger.info("\nTiming Summary")
        logger.info("=" * 60)
        logger.info(f"{'Section':<30} {'Time (s)':<10} {'%':<8}")
        logger.info("-" * 60)
        
        for section in root_sections:
            timing = stats[section]
            logger.info(f"{section:<30} {timing.total_time:>10.3f} {timing.percent_total:>7.1f}%")
        
        logger.info("-" * 60)
        logger.info(f"{'Total':<30} {total_time:>10.3f}")
        logger.info("=" * 60)
