FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "python310", "/bin/bash", "-c"]

RUN pip install --upgrade pip

COPY . .

CMD ["/opt/conda/envs/python310/bin/python", "pipline.py"]