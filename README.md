# 💠 FractalLens: Physics-Informed Edge AI for Rural Diagnostics

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![AWS S3](https://img.shields.io/badge/AWS-S3%20Sync-FF9900.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Edge%20UI-FF4B4B.svg)

**A Hack2skill / AI for Bharat Hackathon Submission** **Team:** The Neural Physicists | **Lead:** Rupak Banerjee

---

## 🚀 The Problem
60% of rural India relies on outdated, low-resolution X-ray and CT machines. Standard Deep Learning models (CNNs) fail on this noisy data, and rural clinics lack the internet bandwidth and hardware to run heavy cloud-based AI.

## 💡 Our Solution: FractalLens
FractalLens is an **Offline-First, Physics-Informed Edge AI**. Instead of relying purely on black-box Deep Learning, we upgrade existing clinic hardware via software. 

We use the **Minkowski-Bouligand Box-Counting algorithm** to extract the Fractal Dimension ($D_f$) of medical scans. Because pathologies (like tumors or pneumonia) are mathematically "rougher" than healthy tissue, this physical signature acts as a powerful pre-filter. We then fuse this mathematical data with a lightweight PyTorch CNN (MobileNetV3) to deliver diagnostic-grade explainability directly on a standard laptop CPU.

### Key Features
* **🔬 Physics-Informed Neural Network (PINN):** Combines Fractal Geometry with Deep Learning for high accuracy on low-res scans.
* **🗺️ Explainable AI:** Generates an Entropy Heatmap (Red = High Fractal Complexity/Anomaly) to guide semi-skilled technicians.
* **📶 Offline-First Edge Database:** Saves patient records and AI inferences locally via SQLite, ensuring zero downtime during internet outages.
* **☁️ AWS Cloud Sync:** Uses `boto3` to securely encrypt and sync the local database to an **Amazon S3 Bucket** whenever internet connectivity is restored, enabling centralized hospital storage and federated learning.

---

## 🛠️ Tech Stack
* **Frontend/Edge UI:** Streamlit
* **Physics Engine:** OpenCV, NumPy, SciPy (Fractal Dimension & Entropy Mapping)
* **Deep Learning Engine:** PyTorch, Torchvision (MobileNetV3 Hybrid)
* **Database & Cloud:** SQLite (Edge), AWS S3 (Cloud), `boto3`

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/Halcyora/AWS-Hackathon-Rupak.git](https://github.com/Halcyora/AWS-Hackathon-Rupak.git)
cd AWS-Hackathon-Rupak
