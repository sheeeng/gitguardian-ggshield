import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional

import click
from pygitguardian.models import (
    AIDiscovery,
    MCPActivityRequest,
    MCPArgumentInfo,
    MCPPromptInfo,
    MCPResourceInfo,
    MCPToolInfo,
)

from ggshield.core.dirs import get_user_home_dir

from ..models import Agent, EventType, HookPayload, HookResult, MCPServer


class Cursor(Agent):
    """Behavior specific to Cursor."""

    @property
    def name(self) -> str:
        return "cursor"

    @property
    def display_name(self) -> str:
        return "Cursor"

    @property
    def config_folder(self) -> Path:
        return get_user_home_dir() / ".cursor"

    def output_result(self, result: HookResult) -> int:
        response = {}
        if result.payload.event_type == EventType.USER_PROMPT:
            response["continue"] = not result.block
            response["user_message"] = result.message
        elif result.payload.event_type == EventType.PRE_TOOL_USE:
            response["permission"] = "deny" if result.block else "allow"
            response["user_message"] = result.message
            response["agent_message"] = result.message
        elif result.payload.event_type == EventType.POST_TOOL_USE:
            pass  # Nothing to do here
        else:
            # Should not happen, but just in case
            click.echo("{}")
            return 2 if result.block else 0

        click.echo(json.dumps(response))
        # We don't use the return 2 convention to make sure our JSON output is read.
        return 0

    def settings_path(self, mode: Literal["local", "global"]) -> Path:
        return Path(".cursor") / "hooks.json"

    @property
    def settings_template(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "hooks": {
                "beforeSubmitPrompt": [{"command": "<COMMAND>"}],
                "preToolUse": [{"command": "<COMMAND>"}],
                "postToolUse": [{"command": "<COMMAND>"}],
            },
        }

    def settings_locate(
        self, candidates: List[Dict[str, Any]], template: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        # We only have one kind of lists: in each hook. Simply look for "ggshield" or "<COMMAND>" in the command.
        for obj in candidates:
            command = obj.get("command", "")
            if "ggshield" in command or "<COMMAND>" in command:
                return obj
        return None

    def project_mcp_file(self, directory: Path) -> Path:
        return directory / ".cursor" / "mcp.json"

    def discover_project_directories(self) -> Iterator[Path]:
        # Because Cursor is based on VS Code, we can reuse the same logic than Copilot.
        user_folder = get_user_home_dir() / ".config" / "Cursor" / "User"
        for file in user_folder.glob("workspaceStorage/*/workspace.json"):
            if (data := self._load_json_file(file)) and "folder" in data:
                path = Path(data["folder"].removeprefix("file://"))
                if path.is_dir():
                    yield path.resolve()

    def discover_capabilities(self, server: MCPServer) -> bool:
        # General Cursor strategy:
        # For each project where Cursor was used, it created a folder with the project name
        # in its configuration folder. Inside that folder, it stores metadata for every
        # MCP server available in that project.
        for configuration in server.configurations:
            # Look for Cursor configurations
            if configuration.agent != self.name:
                continue
            # We need a folder. Note: this also works for user-level configurations.
            # as Cursor will have a `home-<username>` "project".
            if configuration.project is None:
                continue

            # Lookup where Cursor stores the capabilities for the given project.
            mangled = (
                Path(configuration.project).as_posix().replace("/", "-").lstrip("-")
            )
            folder = self.config_folder / "projects" / mangled / "mcps"
            if not folder.exists():
                continue
            # Look for a SERVER_METADATA.json file with the expected name.
            # (each subfolder corresponds to a different MCP server)
            for file in folder.glob("*/SERVER_METADATA.json"):
                metadata = self._load_json_file(file)
                if metadata and metadata.get("serverName") == configuration.name:
                    # Found it! Update the folder
                    folder = file.parent
                    break
            else:
                # We didn't find our MCP server's metadata. Try next configuration.
                continue

            # If we reach this code, we found our MCP server's metadata folder.
            # Hopefully it is connected. If not, Cursor creates a STATUS.md file.
            if (folder / "STATUS.md").exists():
                # Don't go further, we may risk discovering only an "mcp_auth" tool
                # whereas the MCP server may be properly connected in another project.
                continue

            filled = False
            # Tools
            for file in folder.glob("tools/*.json"):
                tool = self._load_json_file(file)
                if not isinstance(tool, dict) or "name" not in tool:
                    continue
                server.tools.append(
                    MCPToolInfo(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        arguments=_parse_tool_arguments(tool.get("arguments")),
                    )
                )
                filled = True
            # Resources
            for file in folder.glob("resources/*.json"):
                resource = self._load_json_file(file)
                if not isinstance(resource, dict) or "uri" not in resource:
                    continue
                server.resources.append(
                    MCPResourceInfo(
                        uri=resource["uri"],
                        name=resource.get("name", ""),
                        description=resource.get("description", ""),
                        mime_type=resource.get("mimeType", ""),
                    )
                )
                filled = True
            # Prompts
            for file in folder.glob("prompts/*.json"):
                prompt = self._load_json_file(file)
                if not isinstance(prompt, dict) or "name" not in prompt:
                    continue
                server.prompts.append(
                    MCPPromptInfo(
                        name=prompt["name"], description=prompt.get("description", "")
                    )
                )
                filled = True
            if filled:
                # Discovery done. Early return.
                return True

        return False

    def parse_mcp_activity(
        self, payload: HookPayload, ai_config: AIDiscovery
    ) -> MCPActivityRequest:
        """Parse the MCP activity from an MCP hook payload."""

        # Cursor only sends the MCP tool, not the server.
        # Fortunately, we should have been able to discover the tools earlier.

        tools_to_server = {}
        for server in ai_config.servers:
            for tool in server.tools:
                # Hopefully we won't have duplicates
                tools_to_server[tool.name] = server.name

        raw_tool_name: str = payload.raw.get("tool_name", "")
        tool_name = raw_tool_name.removeprefix("MCP:")
        server_name = tools_to_server.get(tool_name, "")

        return MCPActivityRequest(
            user=ai_config.user,
            tool=tool_name,
            server=server_name,
            agent=self.name,
            model=payload.raw.get("model", ""),
            cwd=payload.raw.get("workspace_roots", [""])[0],
            input=payload.raw.get("tool_input", {}),
        )


def _parse_tool_arguments(
    schema: Optional[Dict[str, Any]],
) -> Optional[List[MCPArgumentInfo]]:
    """Parse a JSON-Schema ``arguments`` object into a list of MCPArgumentInfo.

    The schema is expected to follow the standard MCP tool descriptor format::

        {"type": "object", "properties": {...}, "required": [...]}
    """
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None
    required_set = set(schema.get("required", []))
    arguments: List[MCPArgumentInfo] = []
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        arguments.append(
            MCPArgumentInfo(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description"),
                required=name in required_set,
            )
        )
    return arguments or None
