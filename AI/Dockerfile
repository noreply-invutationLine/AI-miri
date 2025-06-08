FROM python:3.11-slim

# התקנות מערכת בסיסיות כולל ffmpeg ותלויות ל-moviepy
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# יצירת תיקיית עבודה והעתקת הקבצים
WORKDIR /app
COPY . /app

# שדרוג pip והתקנת תלויות
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# הפקודה שתורץ בזמן הריצה
CMD ["python", "main.py"]
