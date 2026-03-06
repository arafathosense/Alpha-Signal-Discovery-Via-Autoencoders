# Alpha Signal Discovery Via Autoencoder

This project explores the application of deep learning techniques to identify hidden structures within financial signals. Specifically, an **autoencoder neural network** is used to learn latent representations of synthetic financial data. By compressing high-dimensional signals into a lower-dimensional latent space, the model can capture meaningful hidden patterns that may correspond to underlying financial factors.

The project demonstrates the entire workflow, including **data generation, preprocessing, model training, and visualization of learned representations**. The resulting latent factors are visualized in an **interactive three-dimensional space**, enabling intuitive analysis of the relationships between signals.

<img width="900" height="600" alt="newplot" src="https://github.com/user-attachments/assets/774ba6b9-1572-4dd5-81c2-9109936fadbc" />


## Introduction

In quantitative finance and signal processing, identifying hidden factors within large sets of financial signals is a challenging task. Traditional statistical methods often rely on linear models such as Principal Component Analysis (PCA). However, modern **deep learning methods** provide the capability to capture complex nonlinear relationships within the data.

Autoencoders are unsupervised neural networks designed to learn efficient representations of input data by encoding it into a compact latent space and reconstructing it back to the original form. This project utilizes an autoencoder to discover latent structures within synthetic financial signals and visualize them for further interpretation.


## Objectives

The primary objectives of this project are:

* To generate synthetic financial signals for experimentation
* To preprocess and normalize the dataset
* To design and train an autoencoder model using PyTorch
* To extract latent representations learned by the model
* To visualize the latent factors using an interactive 3D visualization


## Methodology

### 1. Synthetic Data Generation

The project begins by generating synthetic financial signals designed to mimic real-world noisy data with underlying hidden structures. This approach allows controlled experimentation while maintaining realistic signal properties.

### 2. Data Preprocessing

Before training the neural network, the generated signals are normalized to ensure stable training and improved convergence of the model.

### 3. Autoencoder Architecture

The autoencoder consists of two main components:

**Encoder**
Compresses the input signals into a lower-dimensional latent representation.

**Decoder**
Reconstructs the original signals from the latent representation.

Through this process, the model learns a compact representation that captures the essential characteristics of the input data.

### 4. Latent Factor Extraction

After training, the encoder portion of the network is used to transform input signals into latent vectors representing hidden factors.

### 5. Visualization

The latent vectors are projected into a **three-dimensional space** and visualized using an interactive Plotly graph. This visualization allows users to explore patterns and clusters within the latent representation space.


## Technologies Used

* **Python**
* **PyTorch**
* **Plotly**
* **scikit-learn**
* **NumPy**
* **pandas**

These tools enable efficient data processing, model training, and visualization.


## Project Structure

```
Alpha-Signal-Discovery-via-Autoencoder/
│
├── main.py
├── latent_factor_3d.html
├── requirements.txt
└── README.md
```

**main.py**
Contains the full implementation of the workflow, including data generation, model training, and visualization.

**latent_factor_3d.html**
Interactive visualization file generated after running the script.

**requirements.txt**
Lists the required Python dependencies.


## Installation

Clone the repository:

```
git clone https://github.com/arafathosense/Alpha-Signal-Discovery-via-Autoencoder.git
```

Navigate to the project directory:

```
cd Alpha-Signal-Discovery-via-Autoencoder
```

Install the required dependencies:

```
pip install torch plotly scikit-learn pandas numpy
```

Alternatively, if a `requirements.txt` file is available:

```
pip install -r requirements.txt
```

## Usage

Run the main script:

```
python main.py
```

After execution, the script will generate an interactive visualization file:

```
latent_factor_3d.html
```

Open this file in a web browser to explore the **3D latent factor space**.


## Example Output

The project produces an interactive 3D visualization where each point represents a signal in the learned latent space. By rotating and zooming the graph, users can explore clusters and relationships between signals.


## Applications

The techniques demonstrated in this project can be extended to several real-world applications, including:

* Quantitative finance and alpha signal discovery
* Feature extraction from high-dimensional data
* Market pattern recognition
* Portfolio signal analysis
* Unsupervised representation learning


## Contribution

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a new feature branch
3. Implement your improvements
4. Submit a pull request

## License

This project is intended for **educational and research purposes**.

## 👤 Author

**HOSEN ARAFAT**  

**Software Engineer, China**  

**GitHub:** https://github.com/arafathosense

**Researcher: Artificial Intelligence, Image Computing, Image Processing, Machine Learning, Deep Learning, Computer Vision**
