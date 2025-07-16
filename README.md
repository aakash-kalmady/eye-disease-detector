_This entire application is encapsulated within a Docker container, ready for deployment._

---

## Technology Stack

| Area                        | Technologies Used               |
| --------------------------- | ------------------------------- |
| **Backend & API**           | Python, Flask                   |
| **Machine Learning**        | TensorFlow, Keras, Scikit-learn |
| **Image Processing**        | OpenCV                          |
| **Containerization**        | Docker                          |
| **Future Cloud Deployment** | AWS EC2                         |

---

## Model Details

#### Dataset

The model was trained on a dataset of over 4000 retinal fundus images, categorized into four classes: Normal, Diabetic Retinopathy, Glaucoma, and Cataract.

#### Preprocessing & Augmentation

To enhance model robustness and prevent overfitting, each image undergoes a preprocessing pipeline using OpenCV:

- Resizing to a uniform input dimension (e.g., 224x224 pixels).
- Normalization of pixel values.
- Data augmentation techniques such as random rotations, flips, and zooms.

#### Training

The CNN model was trained in a Google Colab environment, leveraging GPU acceleration. The complete, documented training process is available in the project repository to ensure transparency and reproducibility.

- **Training Notebook:** `/notebooks/eye_disease_ml.ipynb`
- **Trained Model:** The final trained weights are stored in an H5 file (`/eye_disease_model.h5`).

---

## API Endpoints

### `POST /predict`

Accepts a retinal scan image and returns the model's prediction and confidence score.

- **URL:** `/predict`
- **Method:** `POST`
- **Body:** `multipart/form-data`

  - `file`: The image file to be classified.

- **Success Response (200 OK):**

  ```json
  {
    "prediction": "Diabetic Retinopathy",
    "confidence": 0.92,
    "success": true
  }
  ```

- **Error Responses (400/500):**
  ```json
  {
    "error": "A descriptive error message (e.g., 'No file provided').",
    "success": false
  }
  ```

---
