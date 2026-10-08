FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
# requires artifacts/ from `make data run`
COPY artifacts ./artifacts
ENV ABUSE_DETECT_ARTIFACTS=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "abuse_detect.serve:app", "--host", "0.0.0.0", "--port", "8000"]
