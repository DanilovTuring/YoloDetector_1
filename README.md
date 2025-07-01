LiteLens - Simplified YOLOv5 Object Detection 

LiteLns is a streamlined and developer-friendly obtect detection tool built on top of YOLOv5. it aims to provide clean, readable code for fast prototyping and educational use, while remaining robust enougth for real-world applications.

Features
.🧰 Ligthweigth and minimalistic interface
.🧷 Built on YOLOv5 with PyTorch
.🌐 Easy integrationd and modular design
.📸 Real-Time image inference and annotated outputs
.✅ Beginner-friendly and well-documented codebase

Installation

# 1. Clone the repository
$ git clone https://github.com/DanilovTuring/YoloDetector_1
$ cd litelens

# 2. Create virtual environmnet
$ python -m venv venv
$ source venv/bin/activate

# 3. Intall dependencias
$ pip install -r requeriments.txt

Usage 
# Run detection script
$ python scripts/run.py

intput image: resources/samples/zinedine.jpg
output image: resources/outputs/zinedine_output.jpg
You can replace the image in the samples folder to test with different inputs

License 
MIT License
You are free to use, modify, and distribute it for educational and commercial purposes.

Future Plans
.CLI tool for batch detection
.Webcam/video stream support
.Export annotated results in JSON/CVS 
.Model selector (YOLOv5s/m/l/x)

Credits
Based on the Ultralytics YOLOv5 implementation.




