#  FedMed: Federated Learning for Tuberculosis Detection

Privacy-preserving Tuberculosis (TB) detection using Chest X-ray images with Federated Learning across multiple hospitals.

---

##  Problem Statement

Tuberculosis detection using chest X-rays traditionally depends on expert radiologists, making the process:

* Slow and inconsistent
* Dependent on scarce medical expertise (especially in rural areas)
* Difficult to scale

At the same time:

* Medical data is **highly sensitive and legally protected** under regulations like the IT Act 2000 and DPDP Act 2023
* Hospitals **cannot share raw patient data** externally
* Individual hospitals often have **limited labeled datasets**, reducing model performance

 This makes centralized machine learning **impractical and non-compliant**

---

##  Proposed Solution

FedMed uses **Federated Learning (FL)** to enable collaborative model training across multiple hospitals **without sharing raw data**.

* Each hospital trains a local model on its own dataset
* Only model weights are shared with a central server
* A global model is created using aggregation (FedAvg)
* The updated model is redistributed for further training

This ensures:

*  Data privacy
*  Legal compliance
*  Improved model performance using distributed data

---

##  System Architecture

1. Server initializes a **pretrained ResNet-50 model**
2. Model is sent to multiple hospital clients
3. Each hospital:

   * Trains locally on chest X-ray data
   * Runs for E local epochs
4. Clients send **model weights (NOT data)** back to server
5. Server performs **Federated Averaging (FedAvg)**
6. Updated global model is redistributed
7. Process repeats for multiple rounds (10–20)

---

##  Model Details

The model learns hierarchical features from X-ray images:

* **Low-level features**

  * Edge detection (Sobel-like filters)
  * Texture and contrast patterns

* **Mid-level features**

  * Nodules and infiltrates
  * Lung structure and symmetry

* **High-level features**

  * TB-specific patterns
  * Cavitary lesions
  * Classification boundary (Normal vs TB)

---

##  Tech Stack

* **PyTorch** → Deep learning framework
* **Flower (FLWR)** → Federated Learning framework
* **ResNet-50** → Backbone CNN model
* **Python (Conda environment)** → Environment & dependency management

---

##  My Contributions

* Implemented **Federated Learning pipeline using Flower**
* Simulated **multiple hospital clients locally**
* Integrated **ResNet-50 model for TB classification**
* Handled **training workflow and FL rounds**
* Worked on **data preprocessing and augmentation**
* Ensured **privacy-preserving training (no raw data exchange)**

---

##  Results

* Successfully trained model across **3 simulated hospital nodes**
* Achieved **collaborative learning without data sharing**
* Demonstrated feasibility of **privacy-preserving TB detection**
* Compared **baseline vs federated training performance**

---

##  Privacy & Compliance

* No raw X-ray images are transferred
* Only **model weights** are shared
* Fully aligned with:

  * IT Act 2000
  * DPDP Act 2023

---

##  Setup Instructions

```bash
conda create -n fedmed python=3.10
conda activate fedmed
pip install -r requirements.txt
```

---

##  Team Members

* Uday Ahuja
* Uday Vashist
* Yashasvi Agrawal
* Prachi Hassani

---


##  Future Scope

* Deployment on real hospital infrastructure
* Handling non-IID data more efficiently
* Model personalization per hospital
* Real-time clinical integration

---

##  Key Takeaway

This project demonstrates how **Federated Learning can enable scalable, privacy-preserving AI in healthcare**, solving both **data scarcity and legal constraints** simultaneously.
