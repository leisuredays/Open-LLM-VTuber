FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MCP server files
COPY *.py .
COPY *.json .

# Copy module directories
COPY clients/ ./clients/
COPY modes/ ./modes/

# Default command (can be overridden in docker-compose)
CMD ["python", "time_mcp_server.py"]
