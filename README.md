# AI/ML FHU DEMO

Simple example using tensorflow and keras to determine if water is safe to drink or not

## How to Run

Create a python3 virtual enviroment with whatever you prefer (anaconda, miniconda, python3 -m venv env)

Switch to python enviroment

`pip install requirements.txt`

Run the main program

`python main.py`

## Running with Docker

if you can not get the python enviroment to work or you're lazy and just want to see the program running then you can use docker

Build the docker image

`docker build --pull --rm -f "DockerFile" -t fhudemo:latest "."`

Run the docker container

`docker run -it --rm fhudemo`