"""DateTime tool implementation for retrieving current date and time."""

import datetime
from app.tools.base_tool import BaseTool

class DateTimeTool(BaseTool):
    """Tool for retrieving current date and time information."""

    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return "Retrieve the current date, time, day of the week, or time zone information."

    def can_handle(self, query: str) -> bool:
        return any(kw in query.lower() for kw in ["time", "date", "day", "clock"])

    def available(self) -> bool:
        return True

    def execute(self, query: str) -> dict:
        try:
            local_now = datetime.datetime.now()
            utc_now = datetime.datetime.utcnow()
            
            day_name = local_now.strftime("%A")
            local_str = local_now.strftime("%Y-%m-%d %H:%M:%S")
            utc_str = utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")
            
            # Formulate detailed content
            content_parts = [
                f"Local Time: {local_str}",
                f"UTC Time: {utc_str}",
                f"Day of the Week: {day_name}",
                f"Date: {local_now.strftime('%d %B %Y')}"
            ]
            content = "\n".join(content_parts)

            data = {
                "local_time": local_str,
                "utc_time": utc_str,
                "day_of_week": day_name,
                "date": local_now.strftime("%Y-%m-%d"),
                "formatted_date": local_now.strftime("%d %B %Y")
            }

            return {
                "success": True,
                "tool": self.name,
                "data": data,
                "content": content,
                "sources": [],
                "metadata": {}
            }
        except Exception as exc:
            return {
                "success": False,
                "tool": self.name,
                "error": f"Failed to retrieve current date/time: {exc}",
                "content": "",
                "sources": [],
                "metadata": {}
            }
