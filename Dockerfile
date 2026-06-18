FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
# The dependency list is read straight out of pyproject.toml so there is a
# single source of truth for required packages.
COPY pyproject.toml ./
RUN python3 -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > requirements.txt \
    && pip install -r requirements.txt

COPY . .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/_stcore/health')" || exit 1

CMD ["streamlit", "run", "simple_app.py", "--server.port=5000", "--server.address=0.0.0.0"]
