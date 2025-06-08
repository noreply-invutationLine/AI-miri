# השתמש בדימוי בסיס קל משקל עם פייתון 3.11
FROM python:3.11-slim

# התקנת ספריות מערכת חיוניות להפעלת moviepy ו-ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# הגדרת תיקיית העבודה
WORKDIR /app

# העתקת קבצי הפרויקט לקונטיינר
COPY . /app

# התקנת pip מעודכן וכל התלויות הנדרשות
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# הרצת הסקריפט הראשי
CMD ["python", "main.py"]
