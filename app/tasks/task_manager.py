TASKS = {
    "easy": {
        "description": (
            "Single service outage: the cache layer has gone down, causing elevated "
            "latency across the platform. Restart it to restore normal operation."
        ),
        "initial_state": {
            "services": {
                "auth-service":    "running",
                "payment-service": "running",
                "api-gateway":     "running",
                "database":        "running",
                "cache":           "down",
            },
            "logs": [
                "ERROR  cache           - Connection refused on port 6379",
                "WARN   api-gateway     - Cache miss rate at 100%; falling back to DB",
                "WARN   payment-service - Response latency elevated (320 ms)",
                "INFO   database        - Read load increased due to cache unavailability",
            ],
            "config": {
                "retry_limit":        3,
                "timeout_seconds":    30,
                "alert_threshold":    1,
                "auto_restart":       False,
                "escalation_enabled": True,
            },
        },
        "expected_actions": [
            {"type": "restart_service", "target": "cache"},
        ],
    },

    "medium": {
        "description": (
            "Dual service failure: both the auth-service and the database are down. "
            "The auth-service depends on the database, so restore the database first, "
            "then bring auth-service back online, and tighten the timeout config to "
            "prevent a recurrence."
        ),
        "initial_state": {
            "services": {
                "auth-service":    "down",
                "payment-service": "running",
                "api-gateway":     "running",
                "database":        "down",
                "cache":           "running",
            },
            "logs": [
                "CRITICAL database       - Disk I/O error; service terminated",
                "ERROR  auth-service     - Cannot reach database; shutting down",
                "ERROR  api-gateway      - Authentication unavailable; requests failing",
                "WARN   payment-service  - Transactions queued; auth checks bypassed",
                "INFO   cache            - Operating normally",
                "CRITICAL system         - 2 critical services are down",
            ],
            "config": {
                "retry_limit":        3,
                "timeout_seconds":    30,
                "alert_threshold":    2,
                "auto_restart":       False,
                "escalation_enabled": True,
            },
        },
        "expected_actions": [
            {"type": "restart_service", "target": "database"},
            {"type": "restart_service", "target": "auth-service"},
            {"type": "update_config",   "target": "", "params": {"timeout_seconds": 10, "retry_limit": 5}},
        ],
    },

    "hard": {
        "description": (
            "Cascading failure: a database crash has caused auth-service to fall, "
            "which took api-gateway with it, leaving payment-service degraded and "
            "cache overloaded. Escalate immediately to engage on-call, restore the "
            "database, clear the noise from the logs, restart each dependent service "
            "in dependency order, then enable auto-restart and lower the alert "
            "threshold so the system self-heals faster next time."
        ),
        "initial_state": {
            "services": {
                "auth-service":    "down",
                "payment-service": "down",
                "api-gateway":     "down",
                "database":        "down",
                "cache":           "down",
            },
            "logs": [
                "CRITICAL database       - Storage volume unmounted unexpectedly",
                "CRITICAL database       - Failed to restart; volume still unavailable",
                "ERROR  auth-service     - DB connection pool exhausted; service down",
                "ERROR  api-gateway      - Auth-service unreachable; rejecting all traffic",
                "ERROR  payment-service  - Downstream auth failure; halting transactions",
                "ERROR  cache            - Eviction policy failure; OOM condition reached",
                "CRITICAL system         - All 5 services are down — full outage detected",
                "WARN   system           - PagerDuty alert suppressed (threshold config)",
            ],
            "config": {
                "retry_limit":        1,
                "timeout_seconds":    60,
                "alert_threshold":    5,
                "auto_restart":       False,
                "escalation_enabled": True,
            },
        },
        "expected_actions": [
            {"type": "escalate"},
            {"type": "restart_service", "target": "database"},
            {"type": "clear_logs"},
            {"type": "restart_service", "target": "cache"},
            {"type": "restart_service", "target": "auth-service"},
            {"type": "restart_service", "target": "api-gateway"},
            {"type": "restart_service", "target": "payment-service"},
            {
                "type": "update_config",
                "params": {
                    "auto_restart":    True,
                    "alert_threshold": 1,
                    "retry_limit":     5,
                    "timeout_seconds": 15,
                },
            },
        ],
    },
}


class TaskManager:
    """Thin wrapper around TASKS that integrates with IncidentResponseEnv."""

    LEVELS = ("easy", "medium", "hard")

    def __init__(self):
        self._tasks = TASKS

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_task(self, level: str) -> dict:
        if level not in self._tasks:
            raise ValueError(f"Unknown level '{level}'. Choose from: {self.LEVELS}")
        return self._tasks[level]

    def all_tasks(self) -> dict:
        return dict(self._tasks)

    def describe(self, level: str) -> str:
        return self.get_task(level)["description"]

    # ------------------------------------------------------------------
    # Env integration
    # ------------------------------------------------------------------

    def load_into_env(self, env, level: str):
        """
        Seed an IncidentResponseEnv instance with a task's initial_state.

        Usage:
            env = IncidentResponseEnv()
            tm  = TaskManager()
            state = tm.load_into_env(env, "hard")
        """
        task = self.get_task(level)
        state = task["initial_state"]

        env.reset()
        env._services = dict(state["services"])
        env._logs     = list(state["logs"])
        env._config   = dict(state["config"])

        return env.state()

    def expected_actions(self, level: str) -> list[dict]:
        return list(self.get_task(level)["expected_actions"])

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def evaluate(self, env, level: str) -> dict:
        """
        Run expected_actions against env and return a summary report.

        Returns
        -------
        dict with keys: level, steps, total_reward, solved, log
        """
        self.load_into_env(env, level)
        actions = self.expected_actions(level)

        total_reward = 0.0
        run_log      = []

        for action in actions:
            obs, reward, done, info = env.step(action)
            total_reward += reward
            run_log.append({
                "action": action,
                "reward": reward,
                "done":   done,
                "info":   info,
            })
            if done:
                break

        return {
            "level":        level,
            "steps":        len(run_log),
            "total_reward": round(total_reward, 2),
            "solved":       all(s == "running" for s in obs["services"].values()),
            "log":          run_log,
        }

    def __repr__(self) -> str:
        return f"TaskManager(levels={self.LEVELS})"


# ----------------------------------------------------------------------
# Quick smoke-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    try:
        from incident_response_env import IncidentResponseEnv
        env = IncidentResponseEnv()
        tm  = TaskManager()

        for level in TaskManager.LEVELS:
            result = tm.evaluate(env, level)
            status = "✓ SOLVED" if result["solved"] else "✗ UNSOLVED"
            print(f"[{level.upper():6s}] {status}  |  "
                  f"steps={result['steps']}  reward={result['total_reward']}")
    except ImportError:
        # Standalone mode — just print task metadata
        tm = TaskManager()
        for level in TaskManager.LEVELS:
            task = tm.get_task(level)
            print(f"\n{'='*60}")
            print(f"  {level.upper()}")
            print(f"{'='*60}")
            print(f"  {task['description']}")
            print(f"  Services down : "
                  f"{[k for k,v in task['initial_state']['services'].items() if v=='down']}")
            print(f"  Actions needed: {len(task['expected_actions'])}")
