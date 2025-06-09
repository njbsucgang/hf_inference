#!/bin/bash
service nginx start
exec gunicorn -k uvicorn.workers.UvicornWorker \
    --worker-tmp-dir /dev/shm \
    --preload \
    --threads 4 \
    --bind 0.0.0.0:${PORT} \
    --workers ${WORKERS} \
    --timeout 300 \
    app:app
