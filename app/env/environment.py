class IncidentResponseEnv:
    """
    OpenEnv-compatible environment simulating an incident response scenario.
    Manages services, logs, and config without external dependencies.
    """

    VALID_ACTIONS = {"restart_service", "clear_logs", "update_config", "escalate"}

    def __init__(self):
        self._services: dict[str, str] = {}
        self._logs: list[str] = []
        self._config: dict = {}
        self._step_count: int = 0
        self._max_steps: int = 20
        self._done: bool = False
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Initialise a fresh incident scenario and return the initial state."""
        self._step_count = 0
        self._done = False

        self._services = {
            "auth-service":     "down",
            "payment-service":  "running",
            "api-gateway":      "down",
            "database":         "running",
            "cache":            "running",
        }

        self._logs = [
            "ERROR  auth-service   - Connection timeout after 30 s",
            "ERROR  api-gateway    - Upstream unreachable (auth-service)",
            "WARN   payment-service - Elevated latency detected (450 ms)",
            "INFO   database       - Replication lag within acceptable range",
            "ERROR  auth-service   - Failed to acquire DB connection pool",
            "CRITICAL system       - 2 critical services are down",
        ]

        self._config = {
            "retry_limit":        3,
            "timeout_seconds":    30,
            "alert_threshold":    2,
            "auto_restart":       False,
            "escalation_enabled": True,
        }

        return self.state()

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        Execute one action and advance the environment.

        Parameters
        ----------
        action : dict
            {
                "type": str,          # one of VALID_ACTIONS
                "target": str,        # service name (where applicable)
                "params": dict        # optional extra parameters
            }

        Returns
        -------
        observation : dict   – current environment state
        reward       : float – score for this step
        done         : bool  – episode finished?
        info         : dict  – diagnostic details
        """
        if self._done:
            return self.state(), 0.0, True, {"error": "Episode already finished. Call reset()."}

        self._step_count += 1
        action_type = action.get("type", "")
        target      = action.get("target", "")
        params      = action.get("params", {})

        if action_type not in self.VALID_ACTIONS:
            reward = -1.0
            info   = {"error": f"Unknown action '{action_type}'. Valid: {sorted(self.VALID_ACTIONS)}"}
            return self.state(), reward, self._done, info

        # Dispatch
        if action_type == "restart_service":
            reward, info = self._handle_restart(target)
        elif action_type == "clear_logs":
            reward, info = self._handle_clear_logs()
        elif action_type == "update_config":
            reward, info = self._handle_update_config(params)
        elif action_type == "escalate":
            reward, info = self._handle_escalate()

        # Episode termination conditions
        if self._all_services_running():
            self._done = True
            reward += 10.0
            info["terminal"] = "All services restored — incident resolved."
        elif self._step_count >= self._max_steps:
            self._done = True
            info["terminal"] = "Max steps reached without full resolution."

        info["step"] = self._step_count
        return self.state(), reward, self._done, info

    def state(self) -> dict:
        """Return a snapshot of the current environment state."""
        return {
            "services": dict(self._services),
            "logs":     list(self._logs),
            "config":   dict(self._config),
            "metadata": {
                "step":           self._step_count,
                "done":           self._done,
                "services_down":  self._count_down(),
            },
        }

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_restart(self, target: str) -> tuple[float, dict]:
        if target not in self._services:
            return -0.5, {"error": f"Unknown service '{target}'."}

        previous = self._services[target]
        if previous == "running":
            self._logs.append(f"WARN   {target} - Restart attempted but service already running")
            return -0.2, {"info": f"'{target}' was already running; no-op."}

        # Restart succeeds; dependent services may recover
        self._services[target] = "running"
        self._logs.append(f"INFO   {target} - Service restarted successfully")
        self._recover_dependents(target)

        reward = 2.0 * self._count_down_reduction(previous_down=self._count_down() + 1)
        return reward, {"info": f"'{target}' restarted. Status: running."}

    def _handle_clear_logs(self) -> tuple[float, dict]:
        cleared = len(self._logs)
        self._logs = [f"INFO   system - Log cleared ({cleared} entries removed)"]
        return 0.5, {"info": f"Cleared {cleared} log entries."}

    def _handle_update_config(self, params: dict) -> tuple[float, dict]:
        if not params:
            return -0.1, {"error": "No params provided for update_config."}

        updated = {}
        for key, value in params.items():
            if key in self._config:
                self._config[key] = value
                updated[key] = value
                self._logs.append(f"INFO   system - Config updated: {key}={value}")

        if not updated:
            return -0.2, {"error": "No recognised config keys in params."}

        # Auto-restart bonus: immediately restart downed services
        if self._config.get("auto_restart") and self._count_down() > 0:
            for svc, status in self._services.items():
                if status == "down":
                    self._services[svc] = "running"
                    self._logs.append(f"INFO   {svc} - Auto-restarted via config change")

        return 0.5, {"info": f"Config updated: {updated}"}

    def _handle_escalate(self) -> tuple[float, dict]:
        if not self._config.get("escalation_enabled"):
            return -0.5, {"error": "Escalation is disabled in current config."}

        self._logs.append("CRITICAL system - Incident escalated to on-call team")
        # Escalation fast-tracks recovery of one downed service
        for svc, status in self._services.items():
            if status == "down":
                self._services[svc] = "running"
                self._logs.append(f"INFO   {svc} - Restored by on-call engineer post-escalation")
                break

        return 1.5, {"info": "Incident escalated; on-call engineer engaged."}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _recover_dependents(self, restored_service: str) -> None:
        """Restore services that depend on the just-recovered service."""
        dependencies = {
            "auth-service": ["api-gateway"],
        }
        for dependent in dependencies.get(restored_service, []):
            if self._services.get(dependent) == "down":
                self._services[dependent] = "running"
                self._logs.append(f"INFO   {dependent} - Auto-recovered after {restored_service} came online")

    def _all_services_running(self) -> bool:
        return all(s == "running" for s in self._services.values())

    def _count_down(self) -> int:
        return sum(1 for s in self._services.values() if s == "down")

    def _count_down_reduction(self, previous_down: int) -> int:
        return max(0, previous_down - self._count_down())

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"IncidentResponseEnv("
            f"step={self._step_count}, "
            f"done={self._done}, "
            f"services_down={self._count_down()})"
        )


# ----------------------------------------------------------------------
# Quick smoke-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    env = IncidentResponseEnv()
    print("=== Initial State ===")
    s = env.state()
    print("Services:", s["services"])
    print("Config:  ", s["config"])
    print()

    actions = [
        {"type": "restart_service", "target": "auth-service"},
        {"type": "update_config",   "params": {"auto_restart": True, "timeout_seconds": 15}},
        {"type": "escalate"},
        {"type": "clear_logs"},
    ]

    for action in actions:
        obs, reward, done, info = env.step(action)
        print(f"Action : {action['type']}")
        print(f"Reward : {reward}")
        print(f"Done   : {done}")
        print(f"Info   : {info}")
        print(f"Services: {obs['services']}")
        print()
        if done:
            break
