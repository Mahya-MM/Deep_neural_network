# Deep L-Layer Neural Network for Image Classification  

This repository implements a **Deep L-Layer Neural Network** for **binary image classification** (cat vs. non-cat), which allows users to customize the **number of layers** and **neurons per layer**.  
It is based on the **Deep Learning Specialization by Andrew Ng**.

---

## 📌 Project Overview  

The neural network follows these steps:  
1. **Forward Propagation**: Compute activations for each layer.  
2. **Compute Cost**: Use the cross-entropy loss function.  
3. **Backward Propagation**: Compute gradients for parameter updates.  
4. **Update Parameters**: Apply **gradient descent** for optimization.  
5. **Prediction**: Use trained parameters to classify new images.  

---

## 📂 Repository Structure  

- **`DEEP_L_Layer_NN.ipynb`** → Jupyter Notebook containing the implementation and explanation.  
- **`DNN_app.py`** → Python module containing helper functions for building and training the neural network.  
- **`datasets/train_catvnoncat.h5`** → Training dataset.  
- **`datasets/test_catvnoncat.h5`** → Test dataset.  
---

🤝 Contributing
Feel free to fork the repository, submit issues, or contribute improvements! 🚀
