FROM mmdetection:latest

COPY mosquito_detection_dataset data/mosquito_detection_dataset
COPY combined_test data/combined_test

COPY mmdet mmdet
COPY tools tools

COPY configs configs

ENV DISPLAY=:0

RUN echo '#!/bin/bash\npython3 tools/train.py "$@"' > /usr/bin/train && \
    chmod +x /usr/bin/train

RUN echo '#!/bin/bash\npython3 tools/test.py "$@"' > /usr/bin/test && \
    chmod +x /usr/bin/test

RUN echo '#!/bin/bash\npython3 tools/analysis_tools/analyze_logs.py "$@"' > /usr/bin/analyze_logs && \
    chmod +x /usr/bin/analyze_logs

RUN echo '#!/bin/bash\npython3 tools/analysis_tools/browse_dataset.py "$@"' > /usr/bin/browse_dataset && \
    chmod +x /usr/bin/browse_dataset

RUN echo '#!/bin/bash\npython3 tools/analysis_tools/confusion_matrix.py "$@" --show' > /usr/bin/confusion_matrix && \
    chmod +x /usr/bin/confusion_matrix



ENTRYPOINT [ "/bin/bash" ]



