FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/alphasage:/opt/alphasage/src \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch

RUN apt-get update && apt-get install -y libgomp1 libglib2.0-0 libstdc++6
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5

WORKDIR /opt/qm-service
COPY docker/alphasage/requirements.txt /opt/qm-service/requirements.txt
RUN pip install --no-cache-dir --timeout 120 --retries 5 'torch==2.7.1+cu128' \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --extra-index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir --timeout 120 --retries 5 -r /opt/qm-service/requirements.txt
# AlphaSAGE pins this version; newer joblib releases removed an attribute used
# by Qlib 0.9.6's ParallelExt data loader.
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      joblib==1.3.2 scikit-learn==1.7.1

COPY --from=alphasage_source src/ /opt/alphasage/src/
COPY --from=alphasage_source train_gfn.py /opt/alphasage/train_gfn.py
COPY docker/alphasage/service.py /opt/qm-service/service.py

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin alphasage && \
    mkdir -p /jobs /tmp/matplotlib /tmp/torch && \
    chown -R 10001:10001 /jobs /tmp/matplotlib /tmp/torch

USER 10001:10001
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; request=urllib.request.Request('http://127.0.0.1:8090/health', headers={'Authorization':'Bearer '+os.environ['QM_ALPHASAGE_TOKEN']}); urllib.request.urlopen(request, timeout=3)"
CMD ["python", "-m", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8090", "--no-access-log"]
