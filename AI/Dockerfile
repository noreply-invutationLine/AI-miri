FROM python:3.11-slim

# התקנות מערכת בסיסיות
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# העתקת קבצים לקונטיינר
WORKDIR /app
COPY . /app

# התקנת התלויות של פייתון
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# הרצת הקוד
CMD ["python", "main.py"]
