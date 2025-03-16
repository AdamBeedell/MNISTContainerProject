# Use the official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Run Streamlit when the container starts
CMD ["streamlit", "run", "streamlitUi_one.py"]