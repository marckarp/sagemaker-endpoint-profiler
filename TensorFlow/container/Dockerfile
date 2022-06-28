FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.3.0-cpu
RUN pip3 install codeguru_profiler_agent flask
COPY Files/serve.py  /sagemaker/serve.py
COPY Files/gunicorn_conf.py /sagemaker/gunicorn_conf.py