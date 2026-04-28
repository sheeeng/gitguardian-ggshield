import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional

import click
from pygitguardian.models import AIDiscovery, MCPActivityRequest

from ggshield.core.dirs import get_user_home_dir

from ..models import Agent, EventType, HookPayload, HookResult, MCPConfiguration, Scope


class Claude(Agent):
    """Behavior specific to Claude Code."""

    @property
    def name(self) -> str:
        return "claude-code"

    @property
    def display_name(self) -> str:
        return "Claude Code"

    @property
    def config_folder(self) -> Path:
        return get_user_home_dir() / ".claude"

    def output_result(self, result: HookResult) -> int:
        response = {}
        if result.block:
            if result.payload.event_type in [
                EventType.USER_PROMPT,
                EventType.POST_TOOL_USE,
            ]:
                response["decision"] = "block"
                response["reason"] = result.message
                response["additionalContext"] = result.message
            elif result.payload.event_type == EventType.PRE_TOOL_USE:
                response["hookSpecificOutput"] = {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": result.message,
                }
            else:
                # Should not happen, but just in case use Claude's "universal" fields.
                response = {
                    "continue": False,
                    "stopReason": result.message,
                }
        else:
            response["continue"] = True

        click.echo(json.dumps(response))
        # We don't use the return 2 convention to make sure our JSON output is read.
        return 0

    def settings_path(self, mode: Literal["local", "global"]) -> Path:
        return Path(".claude") / "settings.json"

    @property
    def settings_template(self) -> Dict[str, Any]:
        return {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": ".*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "<COMMAND>",
                            }
                        ],
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": ".*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "<COMMAND>",
                            }
                        ],
                    }
                ],
                "UserPromptSubmit": [
                    {
                        "matcher": ".*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "<COMMAND>",
                            }
                        ],
                    }
                ],
            }
        }

    def settings_locate(
        self, candidates: List[Dict[str, Any]], template: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        # We have two kind of lists: at the root of each hook (with a matcher)
        # and in each hook (with a list of commands).
        if "matcher" in template:
            for obj in candidates:
                if obj.get("matcher") == template["matcher"]:
                    return obj
            return None
        for obj in candidates:
            command = obj.get("command", "")
            if "ggshield" in command or "<COMMAND>" in command:
                return obj
        return None

    def project_mcp_file(self, directory: Path) -> Path:
        return directory / ".mcp.json"

    def _get_user_mcp_configurations(self) -> Iterator[MCPConfiguration]:
        """Look into ~/.claude.json for both user-level and project-level MCP server entries."""
        # Load config file
        filepath = get_user_home_dir() / ".claude.json"
        if not (data := self._load_json_file(filepath)):
            return

        # User-level mcpServers
        yield from self._parse_servers_block(data, Scope.USER, None)

        # Per-project entries in projects dict
        projects = data.get("projects", {})
        if not isinstance(projects, dict):
            return
        for project_key, project_data in projects.items():
            if not isinstance(project_data, dict):
                continue
            yield from self._parse_servers_block(
                project_data, Scope.USER, Path(project_key)
            )

    def discover_project_directories(self) -> Iterator[Path]:
        """Discover project directories by scraping config files."""
        history_file = self.config_folder / "history.jsonl"
        projects = set()
        for line in self._load_jsonl_file(history_file):
            if "project" in line:
                projects.add(Path(line["project"]))
        for project in projects:
            if project.is_dir():
                yield project.resolve()

    def parse_mcp_activity(
        self, payload: HookPayload, ai_config: AIDiscovery
    ) -> MCPActivityRequest:
        """Parse the MCP activity from an MCP hook payload."""

        # Claude Code's hook tool name is "mcp__{server}__{tool}"
        raw_tool_name: str = payload.raw.get("tool_name", "")
        parts = raw_tool_name.split("__")
        # The server name can be anything, but we assume no MCP tool has a "__" in its name
        tool = parts[-1]
        server_cfg_name = "__".join(parts[1:-1])

        # Lookup the server name based on its configuration name
        # Fallback to the server name if not found
        server_name = server_cfg_name
        for server in ai_config.servers:
            for configuration in server.configurations:
                if _mangle_server_name(configuration.name) == server_cfg_name:
                    server_name = server.name
                    break

        return MCPActivityRequest(
            user=ai_config.user,
            tool=tool,
            server=server_name,
            agent=self.name,
            model="",
            cwd=payload.raw.get("cwd", ""),
            input=payload.raw.get("tool_input", {}),
        )


MANGLING_PATTERN = re.compile(r"[^A-Za-z0-9-]")


def _mangle_server_name(name: str) -> str:
    """Mangle a server name in the same way Claude Code does."""
    return MANGLING_PATTERN.sub("_", name)
