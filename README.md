# SPBN(Split-BatchNorm) adversarial training
Adversarial training using split-batch normalization.
To test out the efficiency of "[AdvProp](https://arxiv.org/pdf/1911.09665.pdf)", i just tried simple study.

Specifically, BN normalizes input features by the mean and variance computed within each mini-batch. One intrinsic assumption of utilizing BN is that the input features should come from a single or similar distributions. **This normalization behavior could be problematic if the mini-batch contains data from different distributions**, therefore resulting in inaccurate statistics estimation.
![Screenshot](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_399246%2Fimages%2Fx3.png)


# Objective Function
![equation](https://latex.codecogs.com/gif.latex?L_%7BSBAT%7D%20%3D%20%5Clambda%20%5Ccdot%20L%5E%7Badv%7D_%7BCE%7D%20&plus;%20%281-%5Clambda%29%20%5Ccdot%20L%5E%7Bclean%7D_%7BCE%7D)

We train the model with both adversarial examples and clean images.
Proposed model is differnet in that the model is conveted into the split batchnorm model which computes the mean and variance using independent batchnormalization(also called auxiliary batchnormalization)

# Experiments
We use basic ResNet introduced in Madry Paper. 
Baseline model represents the model who has been trained with adversarial data only and spbn was not applied.

| Accuracy  | Baseline([Madry](https://arxiv.org/pdf/1706.06083.pdf))  | Lambda = 0.5 | Lambda = 0.7 |
| :------------ | :------------| :------------| :------------|
| Clean      | 78.1 |  ||
| FGSM       | 58.46 |  ||
| MI-FGSM 20 | 49.54 |  ||
| PGD 7/40   | 48.09/45.22        |   ||
