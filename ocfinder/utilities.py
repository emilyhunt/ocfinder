"""Utility functions for use across ocfinder."""

import datetime


def print_itertime(start_time: datetime.datetime, completed_steps, total_steps):
    """Utility function for providing estimated finished times and updating at the final verbose step of loops."""
    # Get everything we need
    remaining_steps = total_steps - completed_steps
    time_now = datetime.datetime.now()
    time_per_step = (time_now - start_time) / completed_steps

    # Output it!
    print(f"  step {completed_steps} of {total_steps} is done!")
    print("  time per step: ", time_per_step)

    if completed_steps != total_steps:
        finish_estimate = time_now + time_per_step * remaining_steps
        print("  estimated fin: ", finish_estimate)
    else:
        print("  finished iterating!")
