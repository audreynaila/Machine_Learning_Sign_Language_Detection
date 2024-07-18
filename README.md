# Machine Learning Sign Language Detection

## Application Overview

The "Machine Learning Sign Language Detection" application aims to provide a seamless and accurate solution for sign language recognition. It utilizes a trained neural network model to analyze live video feeds or pre-recorded videos, identifying specific hand gestures and converting them into corresponding letters or words.

### Key Features

- **Real-Time Detection**: Processes live video input from a webcam, detecting and recognizing sign language gestures in real-time.
- **High Accuracy**: Utilizes state-of-the-art machine learning models trained on extensive sign language datasets to ensure high accuracy in gesture recognition.
- **User-Friendly Interface**: Features an intuitive and accessible user interface for easy operation and interaction.
- **Versatile Use Cases**: Suitable for educational purposes, communication aid for the hearing impaired, and as a tool for learning sign language.

### Applications

- **Assistive Technology**: Helping individuals with hearing or speech impairments communicate more effectively.
- **Educational Tools**: Providing an interactive learning tool for people interested in learning sign language.
- **Integration with Other Technologies**: Potential integration with virtual assistants and other AI-driven applications to enhance their accessibility.


## Installation

Make sure you have the following libraries installed:

### Prerequisites

- TensorFlow
- Protobuf
- OpenCV
- Object_detection
- Numpy
- etc
  
## Usage

### Data Collection

1. **Required libraries**: `cv2`, `os`, `time`, `uuid`
2. Open the file named `collectingImg.py`.
3. The sign languages detected are `hello`, `thanks`, `yes`, `no`, and `iloveyou`.
4. Each word will be captured in 20 photos with 1-second intervals.
5. A 5-second pause between each word.

### Initializing Training Data

1. **Required libraries**: `tensorflow`, `protobuf`, `gitpython`, `os`, `object_detection`
2. Follow the installation instructions from [TensorFlow Object Detection API Installation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) up to the "Install the Object Detection API" section.
3. Open the file named `Pre-training.py`.

### Training the Model

1. Navigate to the folder where the TensorFlow directory is stored.
2. Open a terminal window in that directory.
3. Copy and paste the following code into the terminal:

   ```shell
   python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=10000

### Detecting Sign Language

1. **Required libraries**: `os`, `object_detection`, `cv2`, `numpy`, `tensorflow`
2. Ensure the main camera used has good quality.
3. Open the file named `ObjectDetection.py`.
4. Wait a moment for the window to appear.
5. Form hand gestures representing `hello`, `thanks`, `yes`, `no`, and `iloveyou`.
6. To exit the program, press the `q` key on the keyboard.

