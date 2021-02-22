# SPBN(Split-BatchNorm) adversarial training
**This is a not official repository**

Adversarial training using split-batch normalization.
To test out the efficiency of "[AdvProp](https://arxiv.org/pdf/1911.09665.pdf)", i just tried simple study.

Specifically, BN normalizes input features by the mean and variance computed within each mini-batch. One intrinsic assumption of utilizing BN is that the input features should come from a single or similar distributions. **This normalization behavior could be problematic if the mini-batch contains data from different distributions**, therefore resulting in inaccurate statistics estimation.
![Screenshot](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_399246%2Fimages%2Fx3.png)

# SPBN Adversarial Training Algorithm 
This is the pseudo code for SPBN adversarial training.
Note that the model will craft the adversaria examples using auxilary BNs, and update its loss using each BN.
For instance, to update the loss against adversarial examples, the model uses auxiliary BN since the model crafts the examples using auxiliary BN, while updates the loss against clean examples using main BNs.

![Screenshot](https://miro.medium.com/max/3212/1*GrGxUQcu4eXWc-4TGzC7pw.png) 

# Objective Function
![equation](https://latex.codecogs.com/gif.latex?L_%7BSBAT%7D%20%3D%20%5Clambda%20%5Ccdot%20L%5E%7Badv%7D_%7BCE%7D%20&plus;%20%281-%5Clambda%29%20%5Ccdot%20L%5E%7Bclean%7D_%7BCE%7D)

We train the model with both adversarial examples and clean images.
Proposed model is differnet in that the model is conveted into the split batchnorm model which computes the mean and variance using independent batchnormalization(also called auxiliary batchnormalization)

# Experiments
We use basic ResNet introduced in Madry Paper. 
Baseline model represents the model who has been trained with adversarial data only and spbn was not applied.
For training the baseline model, we followed the basic setting in Madry paper.
> **PGD adversarial Training**   
> attack_steps = 7  
> attack_eps = 8.0/255.0  
> attack_lr = 2.0/255.0

| Accuracy  | Baseline([Madry](https://arxiv.org/pdf/1706.06083.pdf))  | Lambda = 0.3 | Lambda = 0.5 | Lambda = 0.7 | Lambda = 0.9 |
| :------------ | :------------| :------------| :------------| :------------| :------------|
| Clean      | 78.1 |  || 82.7 | 78.66|
| FGSM       | 58.46 |  || 55.65 | 56.91 |
| PGD 7/40   | 48.09/45.22 |  41.57/37.93  || 46.31/43.47  |
