"""
Utilities for library generation.
"""

import time
from typing import Dict, Tuple


class ProgressMonitor:
    """
    Progress monitor for distributed simulations using explicit gather-based updates.

    Tracks per-rank progress and displays periodic status summaries.
    Updates occur via `update()` calls — no background threads required.
    """

    def __init__(self, size: int, library_name: str = "Library"):
        """
        Initialize the monitor.

        Parameters
        ----------
        size : int
            Total number of MPI ranks.
        library_name : str, optional
            Descriptive label for output headers.
        """
        self.size = size
        self.library_name = library_name
        self._status: Dict[
            int, Tuple[int, int, bool]
        ] = {}  # rank -> (completed, total, done flag)
        self._start_time = time.time()

    def update(self, rank: int, completed: int, total: int, done: bool = False):
        """
        Update status for a rank.

        Parameters
        ----------
        rank : int
            MPI rank identifier.
        completed : int
            Completed parameter count.
        total : int
            Total assigned parameter count.
        done : bool, optional
            True if the rank has completed its workload.
        """
        self._status[rank] = (completed, total, done)

    def print_status(self):
        """
        Print the current progress summary, including elapsed runtime and per-rank details.
        """
        elapsed = time.time() - self._start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        total_completed = 0
        total_assigned = sum(n for _, n, _ in self._status.values() if n)

        print("\n" + "=" * 60)
        print(f"{self.library_name} Progress Summary [{elapsed_str} elapsed]")
        print(
            f"{'Rank':>4} | {'Progress':>10} | {'Completed':>10} / {'Total':<10} {'Status':>8}"
        )
        print("-" * 60)

        for r in range(self.size):
            lidx, n, done = self._status.get(r, (0, None, False))
            pct = 100.0 * lidx / n if n else 0.0
            total_completed += lidx
            total_display = n if n is not None else "NA"
            status_str = "[DONE]" if done else ""
            print(
                f"{r:>4} | {pct:>9.2f}% | {lidx:>10} / {total_display:<10} {status_str:>8}"
            )

        overall_pct = (
            100.0 * total_completed / total_assigned if total_assigned else 0.0
        )
        print("-" * 60)
        print(
            f"{'Total':>4} | {overall_pct:>9.2f}% | {total_completed:>10} / {total_assigned:<10}"
        )
        print("=" * 60 + "\n")


def sync_progress(mpicomm, mpirank, prog_monitor, completed, total, done_flag):
    """
    Perform a non-blocking progress sync. All ranks send status,
    Rank 0 aggregates and optionally prints.

    Returns
    -------
    bool
        True if global completion achieved, False otherwise.
    """
    status_local = (completed, total, done_flag)
    all_status = mpicomm.gather(status_local, root=0)

    global_done = False
    if mpirank == 0:
        for r, (comp, total_r, done_r) in enumerate(all_status):
            prog_monitor.update(r, comp, total_r, done_r)
        prog_monitor.print_status()

        global_done = all(done_r for _, _, done_r in all_status)

    global_done = mpicomm.bcast(global_done, root=0)
    return global_done
