"""
System Scheduler
Manages periodic tasks: sensor polling, optimization, price fetching, model retraining
"""

import logging
import time
import threading
from typing import Callable, Dict, List
from datetime import datetime


class SystemScheduler:
    """Manages scheduled tasks and background jobs"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.running = False
        self.logger = logging.getLogger("IntelliWarm.Scheduler")
    
    def add_task(
        self,
        name: str,
        func: Callable,
        interval: int,
        args: list = None,
        kwargs: dict = None
    ):
        """
        Register a periodic task
        
        Args:
            name: Task identifier
            func: Function to execute
            interval: Run every N seconds
            args: Positional arguments
            kwargs: Keyword arguments
        """
        self.tasks[name] = {
            "func": func,
            "interval": interval,
            "args": args or [],
            "kwargs": kwargs or {},
            "last_run": None,
            "thread": None,
            "stop_event": threading.Event()
        }
        self.logger.info(f"Registered task: {name} (interval={interval}s)")
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            self.logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.logger.info("Scheduler started")
        
        for name, task in self.tasks.items():
            task["stop_event"].clear()
            thread = threading.Thread(
                target=self._run_task_loop,
                args=(name, task),
                daemon=True
            )
            thread.start()
            task["thread"] = thread
    
    def _run_task_loop(self, name: str, task: Dict):
        """Run a task repeatedly at specified interval"""
        while not task["stop_event"].is_set():
            try:
                task["func"](*task["args"], **task["kwargs"])
                task["last_run"] = datetime.now()
            except Exception as e:
                self.logger.error(f"Task '{name}' failed: {e}")
            
            time.sleep(task["interval"])
    
    def stop(self):
        """Stop all scheduled tasks"""
        self.running = False
        
        for name, task in self.tasks.items():
            task["stop_event"].set()
            if task["thread"]:
                task["thread"].join(timeout=5.0)
        
        self.logger.info("Scheduler stopped")
    
    def get_task_status(self, name: str) -> Dict:
        """Get status of a specific task"""
        if name not in self.tasks:
            return None
        
        task = self.tasks[name]
        return {
            "name": name,
            "interval": task["interval"],
            "last_run": task["last_run"],
            "running": task["thread"] and task["thread"].is_alive()
        }
    
    def get_all_tasks_status(self) -> List[Dict]:
        """Get status of all tasks"""
        return [self.get_task_status(name) for name in self.tasks.keys()]
