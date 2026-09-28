# The bot is stdlib-only, so the runtime image needs nothing but Python.
#
# NOTE: this file is untested - there is no Docker daemon in the environment it
# was written in.  It is deliberately minimal (a single CMD, no build steps) so
# the failure modes are obvious.  If you only need a URL, the equivalent
# process-host start command is:
#
#     python -m bot web --host 0.0.0.0 --port $PORT
#
# which needs no container at all.

FROM python:3.11-slim

# Never buffer stdout, or the platform's log tail shows nothing.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first so the layer cache survives a code change. The bot itself
# has no runtime requirements; requirements.txt is dev extras.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true

COPY bot/ ./bot/
COPY config/ ./config/

# Railway / Render / Heroku / Fly all inject PORT.
ENV PORT=8000
EXPOSE 8000

# $HOST lets you pin the interface; unset means 0.0.0.0.
CMD ["sh", "-c", "python -m bot web --host ${HOST:-0.0.0.0} --port ${PORT}"]
