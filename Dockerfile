# Start with a Python 3.9 base image.
FROM python:3.9

# Set /app as the working directory in the container.
WORKDIR /app

# Copy all files from our current directory into /app in the container.
COPY . /app

# Install dependencies from our requirements file.
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available outside the container, the default for FastAPI.
EXPOSE 8000

# Set an environment variable 'NAME' with the value 'World'.
ENV NAME World

# Start our app using uvicorn when the container launches.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
