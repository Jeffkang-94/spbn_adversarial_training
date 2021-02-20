# SPBN(Split-BatchNorm) adversarial training
Adversarial training using split-batch normalization.
To test out the efficiency of "[AdvProp](https://arxiv.org/pdf/1911.09665.pdf)", i just tried simple study.

Specifically, BN normalizes input features by the mean and variance computed within each mini-batch. One intrinsic assumption of utilizing BN is that the input features should come from a single or similar distributions. **This normalization behavior could be problematic if the mini-batch contains data from different distributions**, therefore resulting in inaccurate statistics estimation.
![Screenshot](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_399246%2Fimages%2Fx3.png)

