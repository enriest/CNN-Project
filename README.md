![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project I | Deep Learning: Image Classification with CNN

## Task Description

Students will build a Convolutional Neural Network (CNN) model to classify images from a given dataset into predefined categories/classes.

## Datasets (pick one!)

1. The dataset for this task is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. You can download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).
2. The second dataset contains about 28,000 medium quality animal images belonging to 10 categories: dog, cat, horse, spyder, butterfly, chicken, sheep, cow, squirrel, elephant. The link is [here](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data).

## Assessment Components

1. **Data Preprocessing**
   - Data loading and preprocessing (e.g., normalization, resizing, augmentation).
   - Create visualizations of some images, and labels.

2. **Model Architecture**
   - Design a CNN architecture suitable for image classification.
   - Include convolutional layers, pooling layers, and fully connected layers.

3. **Model Training**
   - Train the CNN model using appropriate optimization techniques (e.g., stochastic gradient descent, Adam).
   - Utilize techniques such as early stopping to prevent overfitting.

4. **Model Evaluation**
   - Evaluate the trained model on a separate validation set.
   - Compute and report metrics such as accuracy, precision, recall, and F1-score.
   - Visualize the confusion matrix to understand model performance across different classes.

5. **Transfer Learning**
    - Evaluate the accuracy of your model on a pre-trained models like ImagNet, VGG16, Inception... (pick one an justify your choice)
        - You may find this [link](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub) helpful.
        - [This](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) is the Pytorch version.
    - Perform transfer learning with your chosen pre-trained models i.e., you will probably try a few and choose the best one.

5. **Code Quality**
   - Well-structured and commented code.
   - Proper documentation of functions and processes.
   - Efficient use of libraries and resources.

6. **Report**
   - Write a concise report detailing the approach taken, including:
     - Description of the chosen CNN architecture.
     - Explanation of preprocessing steps.
     - Details of the training process (e.g., learning rate, batch size, number of epochs).
     - Results and analysis of models performance.
     - What is your best model. Why?
     - Insights gained from the experimentation process.
   - Include visualizations and diagrams where necessary.
   
 7. **Model deployment**
     - Pick the best model 
     - Build an app using Flask - Can you host somewhere other than your laptop? **+5 Bonus points if you use [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving)**
     - User should be able to upload one or multiples images get predictions including probabilities for each prediction
    

## Evaluation Criteria

- Accuracy of the trained models on the validation set. **30 points**
- Clarity and completeness of the report. **20 points**
- Quality of code implementation. **5 points**
- Proper handling of data preprocessing and models training. **30 points**
- Demonstration of understanding key concepts of deep learning. **5 points**
- Model deployment. **10 points**

 <span style="color:red; weight: bold">**Passing Score is 70 points**</span>.

## Submission Details

- Deadline for submission: end of the week or as communicated by your teaching team.
- Submit the following:
  1. Python code files (`*.py`, `ipynb`) containing the model implementation and training process.
  2. A data folder with 5-10 images to test the deployed model/app if hosted somewhere else other than your laptop (strongly recommended! Not a must have)
  2. A PDF report documenting the approach, results, and analysis.
  3. Any additional files necessary for reproducing the results (e.g., requirements.txt, README.md).
  4. PPT presentation

## Additional Notes

- Students are encourage to experiment with different architectures, hyper-parameters, and optimization techniques.
- Provide guidance and resources for troubleshooting common issues during model training and evaluation.
- Students will discuss their approaches and findings in class during assessment evaluation sessions.

## Deployment Instructions

To deploy this project (Flask app for image classification):

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Download the trained model**
   - Make sure `best_model_cifar10.pth` is present in the project or Deployment folder.

3. **Run the Flask app locally**
   ```bash
   cd Deployment
   python app.py
   ```
   The app will be available at [http://localhost:8000](http://localhost:8000)

4. **Deploy to Railway or another cloud platform**
   - Make sure you have a `Procfile` and `requirements.txt` in the `Deployment` folder.
   - Push your code to GitHub.
   - Connect your repository to [Railway](https://railway.app/) or another PaaS.
   - Set the web service to run:
     ```
     python app.py
     ```
   - The deployed app will be available at the URL provided by the platform.

5. **Usage**
   - Open the app in your browser.
   - Upload an image to get predictions and class probabilities.

For more details, see the comments in `Deployment/app.py` and the project notebooks.

## Repository File Structure

- `Project.ipynb` — Main Jupyter notebook for model development, training, and evaluation on CIFAR-10.
- `Transfer learning.ipynb` — Notebook focused on applying and comparing transfer learning approaches.
- `requirements.txt` — List of all Python dependencies needed to run the project and deployment.
- `README.md` — This file. Project overview, instructions, and documentation.
- `CNN Project Report.pdf` — Final project report in PDF format.
- `CNN Project.pptx` — Project presentation slides.
- `best_model_cifar10.pth` — Saved PyTorch model weights for the best custom CNN.
- `best_transfer_model_efficientnet_b0.pth` — Saved weights for the best transfer learning model (EfficientNet-B0).
- `model_stride.pth` — Reference or pre-trained model weights for comparison.
- `cifar10_data/` — Folder containing the CIFAR-10 dataset and extracted files.
    - `cifar-10-python.tar.gz` — Downloaded CIFAR-10 archive.
    - `cifar-10-batches-py/` — Extracted CIFAR-10 data batches.
- `data/` — (Duplicate or alternative) Folder containing the CIFAR-10 dataset and extracted files.
- `Deployment/` — Folder containing all files for web app deployment:
    - `app.py` — Flask web application for image upload and prediction.
    - `requirements.txt` — Dependencies for deployment (can be a copy or subset of the main requirements).
    - `Procfile` — Specifies the command to run the app on cloud platforms.
    - `railway.toml` — Railway deployment configuration (if using Railway).
    - `best_model_cifar10.pth`, `model_stride.pth` — Model weights for use in the deployed app.

You can use or modify these files as needed for local development, training, or deployment.

THIS IS THE PROJECT DEPLOYED IN RAILWAY: https://cnn-project.up.railway.app
*ACCESS MIGHT BE FINISHED AFTER SOME TIME.

