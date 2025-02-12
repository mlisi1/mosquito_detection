FROM mmdetection:latest

RUN pip install seaborn


RUN git clone https://github.com/open-mmlab/mmyolo.git /mmyolo && \
    cd /mmyolo && \
    mim install --no-cache-dir -e .


RUN echo '#!/bin/bash\npython3 /mmdetection/tools/train.py "$@"' > /usr/bin/train && \
    chmod +x /usr/bin/train

RUN echo '#!/bin/bash\npython3 /mmdetection/tools/test.py "$@"' > /usr/bin/test_nn && \
    chmod +x /usr/bin/test_nn

RUN echo '#!/bin/bash\npython3 /mmdetection/tools/analysis_tools/analyze_logs.py "$@"' > /usr/bin/analyze_logs && \
    chmod +x /usr/bin/analyze_logs

RUN echo '#!/bin/bash\npython3 /mmdetection/tools/analysis_tools/browse_dataset.py "$@"' > /usr/bin/browse_dataset && \
    chmod +x /usr/bin/browse_dataset

RUN echo '#!/bin/bash\npython3 /mmdetection/tools/analysis_tools/confusion_matrix.py "$@"' > /usr/bin/confusion_matrix && \
    chmod +x /usr/bin/confusion_matrix


COPY mosquito_detection_dataset data/mosquito_detection_dataset
COPY previous_work_dataset_converted data/previous_work_dataset_converted
COPY unified_mosquito_dataset data/unified_mosquito_dataset


COPY mmdet mmdet
COPY tools tools

COPY configs configs

ENV DISPLAY=:0


ENTRYPOINT [ "/bin/bash" ]



