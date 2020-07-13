# Binary Classification problem to diagnose COVID19 using chest X-ray images. 
Final project from Deep Learning course in Yonsei University.

## Abstract
In this project, I have decided to use the chest x-ray images as my input data to
determine whether an individual patient is Pneumonia affected. Pneumonia is, by definition, an
infection in one or both lungs, caused by bacteria, viruses, and fungi. Although there are other
conditions that cause such infection other than COVID19 itself (such as ARDS, SARS, smoking, etc.),
testing for Pneumonia is an important step in determining whether the patient has the COVID19
virus or not since it is known to affect the respiratory system. Therefore, I have decided to create a
model that can diagnose whether a patient is Pneumonia affected or not by observing the x-ray
image. Such studies and research can be further used to help the medics diagnose patients with
more accuracy and allow patients to receive the results of their diagnoses more quickly.

## Dataset
Dataset used was collected from Kaggle.
* https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset    
The dataset used in this project is collected from a source that again references
other medical sources that provide diagnosed, individual patient’s x-ray chest images. I have
used images from different categories including: COVID19, ARDS(Acute Respiratory
Distress Syndrome), Streptococcus, SARS(Severe Acute Respiratory Syndrome), which are
merged to represent a single category “Pneumonia”

## Code Explanation
* process_data.py

* model.py

* resnet.py

## Preprocessing


## Model

## Conclusion
For this binary classification problem, the test accuracy topped at around 91-92%, showing model
capability in diagnosing actual patients only by looking at x-ray images. The resulting accuracy and
loss trend both seem ideal, where the test is slightly smaller than the train in accuracy, and the test being slightly bigger than train in loss. Overall, the training and hyper parameter tuning process showed that the neural network is often very prone to overfitting, and tuning the learning rate, model complexity, batch size, and data augmentation all contribute to generalizing the model. Further improvement can be made by adding more data to further generalize the model.

## Reference
* https://github.com/FrancescoSaverioZuppichini/ResNet/
* https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
* https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
