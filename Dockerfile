FROM mmdetection:latest

RUN pip install seaborn



# RUN conda init
# RUN conda create --name yolo python=3.8 -y
# RUN conda init
# RUN bash -c "source ~/.bashrc && conda activate yolo"
# RUN conda activate yolo
# RUN pip install ultralytics


RUN conda init bash
RUN conda create --name yolo python=3.8 -y
RUN bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate yolo && pip install ultralytics"


RUN echo '#!/bin/bash\npython3 tools/train.py "$@"' > /usr/bin/train && \
    chmod +x /usr/bin/train

RUN echo '#!/bin/bash\npython3 tools/test.py "$@"' > /usr/bin/test_nn && \
    chmod +x /usr/bin/test_nn

RUN echo '#!/bin/bash\npython3 tools/analysis_tools/analyze_logs.py "$@"' > /usr/bin/analyze_logs && \
    chmod +x /usr/bin/analyze_logs

RUN echo '#!/bin/bash\npython3 tools/analysis_tools/browse_dataset.py "$@"' > /usr/bin/browse_dataset && \
    chmod +x /usr/bin/browse_dataset

RUN echo '#!/bin/bash\npython3 tools/analysis_tools/confusion_matrix.py "$@" --show' > /usr/bin/confusion_matrix && \
    chmod +x /usr/bin/confusion_matrix


COPY mosquito_detection_dataset data/mosquito_detection_dataset
COPY previous_work_dataset_converted data/previous_work_dataset_converted
COPY unified_mosquito_dataset data/unified_mosquito_dataset

COPY mmdet mmdet
COPY tools tools

COPY configs configs

ENV DISPLAY=:0


ENTRYPOINT [ "/bin/bash" ]



