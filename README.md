# Foundation Model Makes Clustering A Better Initialization For Active Learning
## Abstract
Active learning selects the most informative samples from the unlabelled dataset to annotate in the context of a limited annotation budget. While numerous methods have been proposed for subsequent sample selection based on an initialized model, scant attention has been paid to the indispensable phase of active learning: selecting samples for model initialization. Most of the previous studies resort to random sampling or naive clustering. However, random sampling is prone to fluctuation, and naive clustering suffers from convergence speed, particularly when dealing with high-dimensional data such as imaging data. In this work, we propose to integrate foundation models with clustering methods to select samples for active learning initialization. Foundation models refer to those trained on massive datasets by the self-supervised paradigm and capable of generating informative and compacted embeddings for various downstream tasks. Leveraging these embeddings to replace raw features such as pixel values, clustering quickly converges and identifies better initial samples. For a comprehensive comparison, we included a classic ImageNet-supervised model to acquire embeddings. Experiments on two clinical tasks of image classification and segmentation demonstrated that foundation model-based clustering efficiently pinpointed informative initial samples, leading to models showcasing enhanced performance than the baseline methods. We envisage that this study provides an effective paradigm for future active learning.  
## Citation
Han Yuan and Chuan Hong. Foundation Model Makes Clustering A Better Initialization For Active Learning. arXiv preprint arXiv:2402.02561, 2024.
@article{activeyuan2024,
  title={Foundation Model Makes Clustering A Better Initialization For Active Learning},
  author={Han Yuan and Chuan Hong},
  journal={arXiv},
  year={2024}
}
