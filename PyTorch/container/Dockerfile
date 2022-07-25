FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.9.0-cpu-py38
RUN pip3 install codeguru_profiler_agent 
COPY Files/handler_service.py  /opt/conda/lib/python3.8/site-packages/sagemaker_pytorch_serving_container/handler_service.py