disini banner tapi belum jadi


# Catloris - Machine Learning
Hello, this is machine learning part of Catloris application made by Capstone Team C242-PS149 ✨

# Table of Contents
- [Introduction](https://github.com/rzqh/catloris-ml/tree/main#machine-learning-team)
- [ML Team](https://github.com/rzqh/catloris-ml/tree/main#machine-learning-team)
- [What We Do?](https://github.com/rzqh/catloris-ml/tree/main#what-we-do)
- [What We Use?](https://github.com/rzqh/catloris-ml/tree/main#What-Packages-that-we-use-in-Google-Colab/Jupyter-Notebook)
- [Repositories](https://github.com/rzqh/catloris-ml/tree/main#repositories)
- [Image Classification Model](https://github.com/rzqh/catloris-ml/tree/main#image-classification-model)
- [Recommendation System](https://github.com/rzqh/catloris-ml/tree/main#recommendation-system)
- [Machine Learning Model](https://github.com/rzqh/catloris-ml/tree/main#endpoint#machine-learning-model)

# Machine Learning Team

|  Name | Bangkit ID | Contacts |
| ------------ | ------------ | ------------ |
| Rizqi Hasanuddin	 | M479B4KY3948		 | [Github](https://github.com/rzqh) & [Linkedin](https://www.linkedin.com/in/rizqi-hasanuddin/)  |
| Mohammad Iqbal Maulana	 | M562B4KY2570		 | [Github](https://github.com/Mohammadiqbalmaulana2001) & [Linkedin](https://www.linkedin.com/in/mohammad-iqbal-maulana-91917b253/)  |
| Zalfa Nazhifah Huwaida	 | M006B4KX4602		| [Github](https://github.com/zlfnzhaa) & [Linkedin](https://www.linkedin.com/in/zalfa-nazhifah-huwaida-324a4a327/) |

# What We Do?
We are developing a food classification & recommendation model that suggests suitable food options to users.

# What Packages that we use in Google Colab/Jupyter Notebook?

|   Packages |                                
| :----------------: | 
|    Tensorflow |
|  Keras      |  
| Scikit-Learn |  
| Pandas |  
| Numpy |  
| Matplotlib |  

# Repositories

|   Learning Paths       |                                Link                                              |
| :----------------:     | :----------------------------------------------------------------:               |
|   Organization         |            [Github](https://github.com)                    |
|  Machine Learning      |            [Github](https://github.com/rzqh/catloris-ml/tree/main/)   |
|  Machine Learning API  |        [Github](https://github.com/rzqh/catloris-ml/tree/main/apiCatloris)        |

# Machine Learning Model

![Machine Learning Model](https://www.linkpicture.com/)
## Food Classification Model  

### Overview  
The model is designed for classifying food pictures into 15 categories to calculate their nutritional values. It uses a Sequential Model approach with **TensorFlow** and **Keras API**, leveraging **MobileNetV2** for transfer learning.  

### Model Architecture  
- **Base Model: MobileNetV2**  
  - Transfer learning with pre-trained weights from `ImageNet`.  
  - Fine-tuning enabled for some layers.  
  - The first **100 layers are frozen** to preserve pre-trained features.  

- **Custom Layers**  
  - **Global Average Pooling**: Extracts key features from the MobileNetV2 output.  
  - **Flattening**: Prepares the data for dense layers.  
  - **Dense Layers with L2 Regularization**:  
    - A 1024-unit dense layer with ReLU activation and a Dropout rate of 0.3.  
    - A 512-unit dense layer with ReLU activation and a Dropout rate of 0.3.  
  - **Output Layer**: A dense layer with softmax activation for classification into 15 categories.  

### Model Compilation  
- **Optimizer**: Adam  
  - Learning rate: `0.0001`.  
- **Loss Function**: Categorical Cross-Entropy.  
- **Evaluation Metric**: Accuracy.  


![Machine Learning Model](https://www.linkpicture.com/)

For the second model we use KNN for product based recommendation. The recommendation is based on the Body Mass Index, age, fat level.
