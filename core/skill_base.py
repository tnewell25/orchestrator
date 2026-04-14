"""Skill framework — @tool decorator + Skill base class.

Subclass Skill, decorate async methods with @tool, and get_tools() will
auto-build Claude-compatible JSON schemas from Python type hints.
"""
import inspect
import typing


def tool(description: str = ""):
    def decorator(func):
        func._is_tool = True
        func._tool_description = description or func.__doc__ or ""
        return func

    return decorator


class Skill:
    name: str = ""
    description: str = ""

    def __init__(self):
        if not self.name:
            self.name = self.__class__.__name__.lower().replace("skill", "")

    async def setup(self):
        """Override for async initialization."""
        pass

    def get_tools(self) -> list[dict]:
        tools = []
        for attr_name in dir(self):
            method = getattr(self, attr_name, None)
            if callable(method) and getattr(method, "_is_tool", False):
                tools.append(self._build_schema(attr_name, method))
        return tools

    def get_tool_method(self, tool_name: str):
        parts = tool_name.split("-", 1)
        if len(parts) != 2:
            return None
        method = getattr(self, parts[1], None)
        if method and getattr(method, "_is_tool", False):
            return method
        return None

    def _build_schema(self, method_name: str, method) -> dict:
        sig = inspect.signature(method)
        hints = typing.get_type_hints(method)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            hint = hints.get(param_name, str)
            properties[param_name] = self._hint_to_json(hint)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": f"{self.name}-{method_name}",
            "description": method._tool_description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _hint_to_json(self, hint) -> dict:
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)

        if hint is str:
            return {"type": "string"}
        if hint is int:
            return {"type": "integer"}
        if hint is float:
            return {"type": "number"}
        if hint is bool:
            return {"type": "boolean"}
        if origin is list:
            item = args[0] if args else str
            return {"type": "array", "items": self._hint_to_json(item)}
        if origin is typing.Union and type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return self._hint_to_json(non_none[0])
        return {"type": "string"}
