FROM {image_uri}
RUN pip3 install numpy codeguru_profiler_agent flask
COPY Files/serve.py  /sagemaker/serve.py
COPY Files/gunicorn_conf.py /sagemaker/gunicorn_conf.py