FROM python:3.10-slim

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY examples ./examples
RUN pip install --no-cache-dir -e .[dev]
CMD ["pytest"]
