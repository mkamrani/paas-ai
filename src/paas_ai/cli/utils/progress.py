"""
Progress indicators and utilities for CLI operations.

Provides rich progress bars, spinners, and status indicators
for long-running CLI operations.
"""

import time
from contextlib import contextmanager
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.status import Status

from paas_ai.utils.logging import get_logger

console = Console()
logger = get_logger("paas_ai.cli.progress")


@contextmanager
def spinner(message: str, success_message: Optional[str] = None) -> Iterator[Status]:
    """
    Create a spinner for long-running operations.
    
    Args:
        message: Message to display while spinning
        success_message: Message to display on success
    
    Example:
        with spinner("Loading configuration..."):
            time.sleep(2)  # Your operation here
    """
    status = Status(f"⏳ {message}", console=console, spinner="dots")
    status.start()
    
    try:
        yield status
        if success_message:
            console.print(f"✅ {success_message}")
    except Exception:
        console.print(f"❌ Failed: {message}")
        raise
    finally:
        status.stop()


@contextmanager
def progress_bar(
    description: str = "Processing...",
    total: Optional[int] = None,
    show_percentage: bool = True,
    show_time: bool = True,
) -> Iterator[Progress]:
    """
    Create a rich progress bar for operations with known progress.
    
    Args:
        description: Description of the operation
        total: Total number of items (if known)
        show_percentage: Whether to show percentage
        show_time: Whether to show time remaining
    
    Example:
        with progress_bar("Processing files", total=10) as progress:
            task_id = progress.add_task("Processing", total=10)
            for i in range(10):
                time.sleep(0.5)
                progress.update(task_id, advance=1)
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
    ]
    
    if show_percentage:
        columns.append(TaskProgressColumn())
    
    if total is not None:
        columns.append(MofNCompleteColumn())
    
    if show_time:
        columns.append(TimeRemainingColumn())
    
    with Progress(*columns, console=console) as progress:
        yield progress


class ProgressTracker:
    """
    Advanced progress tracker for complex operations with multiple steps.
    """
    
    def __init__(self, description: str):
        self.description = description
        self.steps: list[str] = []
        self.current_step = 0
        self.progress: Optional[Progress] = None
        self.task_id: Optional[int] = None
        
    def add_step(self, step_description: str):
        """Add a step to the progress tracker."""
        self.steps.append(step_description)
        
    def start(self):
        """Start the progress tracking."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            console=console,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            self.description, 
            total=len(self.steps)
        )
        
    def next_step(self, step_description: Optional[str] = None):
        """Move to the next step."""
        if self.progress and self.task_id is not None:
            if step_description:
                self.progress.update(
                    self.task_id, 
                    description=f"{self.description}: {step_description}",
                    advance=1
                )
            else:
                if self.current_step < len(self.steps):
                    self.progress.update(
                        self.task_id,
                        description=f"{self.description}: {self.steps[self.current_step]}",
                        advance=1
                    )
            self.current_step += 1
            
    def finish(self, success_message: Optional[str] = None):
        """Finish the progress tracking."""
        if self.progress:
            self.progress.stop()
            if success_message:
                console.print(f"✅ {success_message}")
            
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finish()
        else:
            self.finish()
            console.print(f"❌ {self.description} failed")


def show_success(message: str):
    """Show a success message with emoji."""
    console.print(f"✅ {message}", style="bold green")


def show_error(message: str):
    """Show an error message with emoji."""
    console.print(f"❌ {message}", style="bold red")


def show_warning(message: str):
    """Show a warning message with emoji."""
    console.print(f"⚠️ {message}", style="bold yellow")


def show_info(message: str):
    """Show an info message with emoji."""
    console.print(f"ℹ️ {message}", style="bold blue")


# Example usage functions for testing
def example_spinner_usage():
    """Example of using the spinner context manager."""
    with spinner("Loading configuration...", "Configuration loaded successfully"):
        time.sleep(2)


def example_progress_bar_usage():
    """Example of using the progress bar."""
    with progress_bar("Processing files", total=10) as progress:
        task_id = progress.add_task("Processing", total=10)
        for i in range(10):
            time.sleep(0.1)
            progress.update(task_id, advance=1)


def example_progress_tracker_usage():
    """Example of using the advanced progress tracker."""
    tracker = ProgressTracker("Setting up environment")
    tracker.add_step("Loading configuration")
    tracker.add_step("Initializing database")
    tracker.add_step("Starting services")
    tracker.add_step("Running health checks")
    
    with tracker:
        for _ in range(4):
            time.sleep(0.5)
            tracker.next_step()


if __name__ == "__main__":
    # Demo all progress utilities
    console.print("\n🚀 Progress Utilities Demo\n", style="bold magenta")
    
    console.print("1. Spinner Example:", style="bold")
    example_spinner_usage()
    
    console.print("\n2. Progress Bar Example:", style="bold")
    example_progress_bar_usage()
    
    console.print("\n3. Progress Tracker Example:", style="bold")
    example_progress_tracker_usage()
    
    console.print("\n4. Message Examples:", style="bold")
    show_success("Operation completed successfully")
    show_warning("This is a warning message")
    show_error("This is an error message") 
    show_info("This is an info message") 