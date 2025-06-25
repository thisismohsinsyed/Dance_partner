# Dance Partner Pose Detection

A real-time human pose estimation system leveraging [MediaPipe](https://google.github.io/mediapipe/solutions/pose.html) and OpenCV, designed for interactive applications such as dance partner analysis, movement studies, and educational tools.

## Features

- **Real-time pose detection** using a standard webcam.
- Utilizes [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) for robust and efficient skeletal tracking.
- Easily extensible for downstream tasks (action recognition, feedback systems, pose-based games, etc.).
- Modular, object-oriented code structure for integration into larger projects.

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/Dance_partner.git
    cd Dance_partner
    ```

2. **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    If `requirements.txt` is missing, install manually:
    ```bash
    pip install opencv-python mediapipe numpy
    ```

---

## Usage

Simply run:

```bash
python pose_partner.py
