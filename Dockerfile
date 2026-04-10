# Use the official TensorFlow image as the base
FROM tensorflow/tensorflow:2.14.0

# Set the working directory in the container
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .


# Command to run your script
CMD ["python", "-u", "-m", "scripts.models-calibration"]
