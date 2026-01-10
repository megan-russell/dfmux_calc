FROM python:3.8-slim

# Install required packages including libngspice-dev for development files
RUN apt-get update && apt-get install -y \
    libngspice0 \
    libngspice0-dev \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link to make sure libngspice.so is in library path
RUN ln -s /usr/lib/x86_64-linux-gnu/libngspice.so.0 /usr/lib/libngspice.so

# Install Python packages
RUN pip install PySpice==1.4.3 nevergrad scipy numpy matplotlib

WORKDIR /app/PATH_TO_WORKING_DIR
COPY . /app

CMD ["python", "SCRIPT_THAT_USES_DFMUX_CALC.py"]
