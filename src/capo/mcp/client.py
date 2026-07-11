import asyncio
import json
import logging
import re
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # expects OPENAI_API_KEY in your .env

_MCP_SERVER_DIR = Path(__file__).parent / "server"
DOCKER_SERVER_PATH = _MCP_SERVER_DIR / "docker_repl_mcp_server.py"
LOCAL_SERVER_PATH = _MCP_SERVER_DIR / "local_repl_mcp_server.py"
LAMBDA_SERVER_PATH = _MCP_SERVER_DIR / "lambda_mcp_server.py"

logging.basicConfig(level=logging.INFO)


def mcp_result_to_string(result: Any) -> str:
    """
    Convert MCP CallToolResult (or similar) into a plain string for OpenAI tool output.
    Handles lists of TextContent and other MCP content blocks.
    """
    content = getattr(result, "content", result)

    if isinstance(content, str):
        return content

    # MCP typically returns a list of content blocks
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if hasattr(item, "text"):  # TextContent
                parts.append(item.text)
            elif hasattr(item, "data"):  # other block types
                parts.append(str(item.data))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()

    if isinstance(content, (dict, int, float, bool)) or content is None:
        return json.dumps(content, ensure_ascii=False)

    return str(content)


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI()
        self.available_tools: list[dict[str, Any]] = []
        self.history: list[dict[str, str]] = []
        self.last_code_snippet: Optional[str] = None
        self.last_saved_path: Optional[str] = None
        self.last_read_path: Optional[str] = None
        self.last_read_content: Optional[str] = None
        self.active_local_session_id: Optional[str] = None
        self.active_docker_session_id: Optional[str] = None
        self.active_lambda_session_id: Optional[str] = None
        self.allow_new_docker_session: bool = False
        self.active_container_id: Optional[str] = None
        self.last_running_containers: list[dict[str, str]] = []
        self.last_created_container_id: Optional[str] = None
        self.last_container_created_at: Optional[float] = None

    def _extract_last_code_block(self, text: str) -> Optional[str]:
        if not text:
            return None

        python_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL)
        if python_blocks:
            return python_blocks[-1].strip()

        any_blocks = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
        if any_blocks:
            return any_blocks[-1].strip()

        return None

    def _should_start_new_docker_session(self, query: str) -> bool:
        text = query.lower()
        keywords = [
            "new session",
            "new container",
            "restart",
            "reset",
            "fresh session",
            "fresh container",
            "new docker",
        ]
        return any(k in text for k in keywords)

    def _has_tool(self, name: str) -> bool:
        return any(t.get("name") == name for t in self.available_tools)

    def _is_create_container_request(self, query: str) -> bool:
        text = query.lower()
        if "new container" in text and not any(k in text for k in ["create", "start"]):
            return False

        return any(
            phrase in text
            for phrase in [
                "create a container",
                "create container",
                "start a new container",
                "start new container",
                "create a new container",
                "create new container",
                "start container",
            ]
        )

    def _extract_image_from_query(self, query: str) -> Optional[str]:
        text = query.lower()
        if "autoimmunolab" in text:
            return "autoimmunolab-repl:latest"
        match = re.search(r"([\w./-]+:[\w.-]+)", text)
        if match:
            return match.group(1)
        return None

    def _find_container_in_query(self, query: str) -> Optional[str]:
        text = query.lower()
        if not self.last_running_containers:
            return None

        for container in self.last_running_containers:
            cid = (container.get("id") or "").lower()
            name = (container.get("name") or "").lower()
            if cid and cid in text:
                return container.get("id")
            if name and name in text:
                return container.get("name")

        return None

    async def _refresh_running_containers(self) -> None:
        if not self._has_tool("list_running_containers") or not self.session:
            return
        tool_result = await self.session.call_tool("list_running_containers", {})
        tool_output_str = mcp_result_to_string(tool_result)
        try:
            payload = json.loads(tool_output_str)
            if payload.get("ok") and isinstance(payload.get("containers"), list):
                self.last_running_containers = payload.get("containers", [])
        except Exception:
            pass

    async def _ensure_active_container(self) -> None:
        if self.active_container_id or not self.session:
            return
        if not self._has_tool("list_running_containers") or not self._has_tool(
            "set_active_container"
        ):
            return

        await self._refresh_running_containers()
        if not self.last_running_containers:
            return

        # docker ps default order is most recent first
        candidate = self.last_running_containers[0]
        target = candidate.get("id") or candidate.get("name")
        if not target:
            return

        try:
            tool_result = await self.session.call_tool(
                "set_active_container", {"container_id_or_name": target}
            )
            tool_output_str = mcp_result_to_string(tool_result)
            payload = json.loads(tool_output_str)
            if payload.get("ok"):
                self.active_container_id = payload.get("active_container")
        except Exception:
            pass

    async def _auto_select_container(self, query: str) -> None:
        if not self._has_tool("set_active_container"):
            return

        text = query.lower()
        id_match = re.search(r"\b[a-f0-9]{12,64}\b", text)
        likely_container = bool(
            "container" in text
            or "containers" in text
            or id_match
            or re.search(r"\b[a-z]+_[a-z]+\b", text)
        )
        if not likely_container:
            return

        if not self.last_running_containers:
            await self._refresh_running_containers()

        target = self._find_container_in_query(query)
        if not target and id_match:
            target = id_match.group(0)

        if not target or target == self.active_container_id:
            return

        try:
            tool_result = await self.session.call_tool(
                "set_active_container", {"container_id_or_name": target}
            )
            tool_output_str = mcp_result_to_string(tool_result)
            payload = json.loads(tool_output_str)
            if payload.get("ok"):
                self.active_container_id = payload.get("active_container")
        except Exception:
            pass

    async def _ensure_container_for_save(self, query: str) -> None:
        if not self.session or not self._has_tool("start_docker_repl_session"):
            return

        text = query.lower()
        wants_container = "container" in text and "save" in text
        if not wants_container:
            return

        await self._refresh_running_containers()
        if self.last_running_containers:
            return

        if self.last_created_container_id and self.last_container_created_at:
            if time.time() - self.last_container_created_at < 30:
                return

        try:
            tool_result = await self.session.call_tool(
                "start_docker_repl_session",
                {"image": "autoimmunolab-repl:latest", "depth": 1},
            )
            tool_output_str = mcp_result_to_string(tool_result)
            payload = json.loads(tool_output_str)
            if payload.get("ok"):
                self.active_container_id = payload.get("container_id")
                if not self.active_docker_session_id:
                    self.active_docker_session_id = payload.get("session_id")
                self.last_created_container_id = payload.get("container_id")
                self.last_container_created_at = time.time()
        except Exception:
            pass

    def _system_instructions(self) -> str:
        return (
            "You are a coding agent that can call MCP tools to save/read files and execute code. "
            "Prefer tools over describing actions. When generating code, if the user does not explicitly "
            "say where to save it, ask a single question offering: save locally, save in container, or do not save. "
            "If the user asks to save, call the appropriate file tool and confirm the path. "
            "Use write_local_file/read_local_file/list_local_files for workspace files, and "
            "docker_repl_write_file/docker_upload_local_file for container files. "
            "Use relative paths for workspace files. "
            "If asked about a file's contents, call read_local_file before answering. "
            "If asked to run a file, use local_repl_execute_file or docker_repl_execute_file rather than retyping code from memory. "
            "If a Docker session already exists, reuse it for all docker_* calls and do not start a new session unless the user explicitly asks for a new container/session. "
            "Do not install Python packages inside the container; assume the Docker image already contains required packages. "
            "When running a script with CLI args in Docker, use docker_repl_run_command with a command list, not shell magic. "
            "When modifying a container file, read it first, then write the full updated content in a single docker_repl_write_file call. "
            "To report active sessions/containers/images, use list_running_containers and list_docker_images. "
            "To sort containers by modification time, call list_running_containers(sort_by='modified'). "
            "When the user names a specific container, call set_active_container and use container_* tools for file operations and commands on that container. "
            "Docker REPL sessions are tracked per server process; if none are listed but containers are running, use container_* tools and do not claim 'no sessions' for the containers. "
            "If an active container is set but no docker REPL session is active, use container_* tools rather than starting a new REPL session. "
            "To export files from the active container to the host workspace, use container_export. "
            "To create a new container, call start_docker_repl_session; do not use container_run_command to run docker CLI."
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server (python or node) over stdio."""
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()

        # Fetch tools once and cache them
        tools_response = await self.session.list_tools()
        tools = tools_response.tools
        print("\nConnected to server with tools:", [t.name for t in tools])

        # IMPORTANT: Responses API tool shape expects name at top level
        self.available_tools = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            }
            for t in tools
        ]
        logging.info("Cached tools: %s", [t["name"] for t in self.available_tools])
        await self._ensure_active_container()

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI + MCP tools (low token budget)."""
        if not self.session:
            raise RuntimeError(
                "Not connected to an MCP server. Call connect_to_server() first."
            )

        if self._is_create_container_request(query) and self._has_tool(
            "start_docker_repl_session"
        ):
            self.allow_new_docker_session = True
            if self.active_docker_session_id:
                try:
                    await self.session.call_tool(
                        "stop_docker_repl_session",
                        {"session_id": self.active_docker_session_id},
                    )
                except Exception:
                    pass

            image = self._extract_image_from_query(query) or "autoimmunolab-repl:latest"
            tool_result = await self.session.call_tool(
                "start_docker_repl_session", {"image": image, "depth": 1}
            )
            tool_output_str = mcp_result_to_string(tool_result)
            try:
                payload = json.loads(tool_output_str)
            except Exception:
                payload = None

            if payload and payload.get("ok"):
                self.active_docker_session_id = payload.get("session_id")
                if payload.get("container_id"):
                    self.active_container_id = payload.get("container_id")
                self.last_created_container_id = payload.get("container_id")
                self.last_container_created_at = time.time()
                response_text = (
                    f"Started new container {payload.get('container_id')} "
                    f"using image {payload.get('image')}."
                )
            else:
                response_text = (
                    f"Failed to start a new container: "
                    f"{payload.get('error') if payload else tool_output_str}"
                )

            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": response_text})
            return response_text

        lower_query = query.lower()
        if (
            "container" in lower_query
            and ("modified" in lower_query or "created" in lower_query)
            and self._has_tool("list_running_containers")
        ):
            sort_by = "modified" if "modified" in lower_query else "created"
            tool_result = await self.session.call_tool(
                "list_running_containers", {"sort_by": sort_by}
            )
            tool_output_str = mcp_result_to_string(tool_result)
            try:
                payload = json.loads(tool_output_str)
            except Exception:
                payload = None

            if (
                payload
                and payload.get("ok")
                and isinstance(payload.get("containers"), list)
            ):
                self.last_running_containers = payload.get("containers", [])
                lines = []
                for idx, container in enumerate(self.last_running_containers, start=1):
                    name = container.get("name") or "unknown"
                    cid = container.get("id") or "unknown"
                    created = container.get("created_at") or "unknown"
                    mtime_iso = container.get("workspace_mtime_iso") or "unknown"
                    if sort_by == "modified":
                        lines.append(
                            f"{idx}. {name} (ID: {cid}) - Last modified at {mtime_iso}"
                        )
                    else:
                        lines.append(
                            f"{idx}. {name} (ID: {cid}) - Created at {created}"
                        )

                response_text = f"Active containers sorted by {sort_by}:\n" + "\n".join(
                    lines
                )
                self.history.append({"role": "user", "content": query})
                self.history.append({"role": "assistant", "content": response_text})
                return response_text

        self.allow_new_docker_session = self._should_start_new_docker_session(query)
        await self._ensure_container_for_save(query)
        await self._ensure_active_container()
        await self._auto_select_container(query)

        # 1) First model call with the user query
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": self._system_instructions()}
                ],
            }
        ]

        if self.last_code_snippet:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Last code snippet:\n```python\n{self.last_code_snippet}\n```",
                        }
                    ],
                }
            )

        if self.last_saved_path:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Last saved file: {self.last_saved_path}",
                        }
                    ],
                }
            )

        if self.last_read_path and self.last_read_content is not None:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Last file read: {self.last_read_path}\n"
                                f"Content:\n{self.last_read_content}"
                            ),
                        }
                    ],
                }
            )

        if self.active_docker_session_id:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Active Docker session: {self.active_docker_session_id}",
                        }
                    ],
                }
            )

        if self.active_local_session_id:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Active local session: {self.active_local_session_id}",
                        }
                    ],
                }
            )

        if self.active_container_id:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Active container is set; use container_* tools for that container.",
                        }
                    ],
                }
            )

        if self.active_container_id:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Active container: {self.active_container_id}",
                        }
                    ],
                }
            )

        for item in self.history[-6:]:
            content_type = (
                "input_text" if item["role"] != "assistant" else "output_text"
            )
            messages.append(
                {
                    "role": item["role"],
                    "content": [{"type": content_type, "text": item["content"]}],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": query}],
            }
        )

        response = await self.openai.responses.create(
            model="gpt-4.1-mini",
            input=messages,
            tools=self.available_tools,
            max_output_tokens=800,  # keep this low since we'll do multiple calls in a loop, and we want to leave room for tool outputs but has to be high enough to get a response that includes tool calls if needed
        )

        final_parts: list[str] = []
        if getattr(response, "output_text", None):
            final_parts.append(response.output_text)

        # 2) Tool loop
        while True:
            # Responses API returns items in response.output
            function_calls = [
                item
                for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]
            if not function_calls:
                break

            for call in function_calls:
                tool_name = call.name
                raw_args = call.arguments  # usually a JSON string

                # Parse tool args robustly
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        tool_args = {"_raw": raw_args}
                else:
                    tool_args = raw_args or {}

                if isinstance(tool_args, dict) and "_raw" in tool_args:
                    raw_payload = tool_args.get("_raw")
                    if isinstance(raw_payload, str):
                        try:
                            parsed = json.loads(raw_payload)
                        except json.JSONDecodeError:
                            parsed = None
                        if isinstance(parsed, dict):
                            tool_args = parsed

                logging.info("Calling MCP tool: %s args=%s", tool_name, tool_args)

                if (
                    tool_name.startswith("docker_")
                    and tool_name != "start_docker_repl_session"
                ):
                    if self.active_docker_session_id:
                        if "session_id" not in tool_args:
                            tool_args["session_id"] = self.active_docker_session_id
                        elif (
                            tool_args["session_id"] != self.active_docker_session_id
                            and not self.allow_new_docker_session
                        ):
                            tool_args["session_id"] = self.active_docker_session_id

                if (
                    tool_name.startswith("local_repl_")
                    and tool_name != "start_local_repl_session"
                ):
                    if self.active_local_session_id:
                        if "session_id" not in tool_args:
                            tool_args["session_id"] = self.active_local_session_id
                        elif tool_args["session_id"] != self.active_local_session_id:
                            tool_args["session_id"] = self.active_local_session_id

                if (
                    tool_name.startswith("lambda_")
                    and tool_name != "lambda_start_session"
                ):
                    if self.active_lambda_session_id:
                        if "session_id" not in tool_args:
                            tool_args["session_id"] = self.active_lambda_session_id
                        elif tool_args["session_id"] != self.active_lambda_session_id:
                            tool_args["session_id"] = self.active_lambda_session_id

                if tool_name.startswith("container_"):
                    if self.active_container_id:
                        if "container_id_or_name" not in tool_args:
                            tool_args["container_id_or_name"] = self.active_container_id
                        elif (
                            tool_args["container_id_or_name"]
                            != self.active_container_id
                        ):
                            tool_args["container_id_or_name"] = self.active_container_id

                if tool_name == "start_docker_repl_session":
                    if (
                        self.active_docker_session_id
                        and not self.allow_new_docker_session
                    ):
                        tool_output_str = json.dumps(
                            {
                                "ok": False,
                                "error": (
                                    "Active Docker session exists: "
                                    f"{self.active_docker_session_id}. "
                                    "Ask user whether to reuse it or start a new session."
                                ),
                            },
                            ensure_ascii=False,
                        )
                    else:
                        if (
                            self.active_docker_session_id
                            and self.allow_new_docker_session
                        ):
                            try:
                                await self.session.call_tool(
                                    "stop_docker_repl_session",
                                    {"session_id": self.active_docker_session_id},
                                )
                            except Exception:
                                pass

                        tool_result = await self.session.call_tool(tool_name, tool_args)
                        tool_output_str = mcp_result_to_string(tool_result)
                elif tool_name == "docker_repl_write_file":
                    if "relative_path" not in tool_args or "content" not in tool_args:
                        tool_output_str = json.dumps(
                            {
                                "ok": False,
                                "error": (
                                    "Malformed docker_repl_write_file args. "
                                    "Required: session_id, relative_path, content."
                                ),
                            },
                            ensure_ascii=False,
                        )
                    else:
                        tool_result = await self.session.call_tool(tool_name, tool_args)
                        tool_output_str = mcp_result_to_string(tool_result)
                else:
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    tool_output_str = mcp_result_to_string(tool_result)

                if tool_name == "start_docker_repl_session":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.active_docker_session_id = payload.get("session_id")
                            if payload.get("container_id"):
                                self.active_container_id = payload.get("container_id")
                    except Exception:
                        pass

                if tool_name == "start_local_repl_session":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.active_local_session_id = payload.get("session_id")
                    except Exception:
                        pass

                if tool_name == "lambda_start_session":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.active_lambda_session_id = payload.get("session_id")
                    except Exception:
                        pass

                if tool_name == "lambda_disconnect":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.active_lambda_session_id = None
                    except Exception:
                        pass

                if tool_name == "set_active_container":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.active_container_id = payload.get("active_container")
                    except Exception:
                        pass

                if tool_name in {
                    "write_local_file",
                    "docker_repl_write_file",
                    "docker_upload_local_file",
                }:
                    try:
                        payload = json.loads(tool_output_str)
                        self.last_saved_path = (
                            payload.get("path")
                            or payload.get("container_path")
                            or payload.get("host_path")
                            or self.last_saved_path
                        )
                    except Exception:
                        pass

                if tool_name == "read_local_file":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.last_read_path = payload.get("path")
                            self.last_read_content = payload.get("content")
                    except Exception:
                        pass

                if tool_name == "docker_repl_read_file":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.last_read_path = payload.get("container_path")
                            self.last_read_content = payload.get("content")
                    except Exception:
                        pass

                if tool_name == "stop_docker_repl_session":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.active_docker_session_id = None
                    except Exception:
                        pass

                if tool_name == "stop_local_repl_session":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.active_local_session_id = None
                    except Exception:
                        pass

                if tool_name == "get_active_container":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok"):
                            self.active_container_id = payload.get("active_container")
                    except Exception:
                        pass

                if tool_name == "list_running_containers":
                    try:
                        payload = json.loads(tool_output_str)
                        if payload.get("ok") and isinstance(
                            payload.get("containers"), list
                        ):
                            self.last_running_containers = payload.get("containers", [])
                    except Exception:
                        pass

                # Send tool output back to OpenAI, continuing the same response thread
                response = await self.openai.responses.create(
                    model="gpt-4.1-mini",
                    previous_response_id=response.id,
                    input=[
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": tool_output_str,
                        }
                    ],
                    tools=self.available_tools,
                    max_output_tokens=800,
                )

                if getattr(response, "output_text", None):
                    final_parts.append(response.output_text)

        final_text = "\n".join(p for p in final_parts if p).strip()

        if final_text:
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": final_text})

            code = self._extract_last_code_block(final_text)
            if code:
                self.last_code_snippet = code

        return final_text

    async def chat_loop(self):
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                answer = await self.process_query(query)
                print("\n" + answer)
            except Exception as e:
                print(f"\nError: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
