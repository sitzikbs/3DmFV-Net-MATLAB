This is the code for training a point cloud classification network using 3D modified Fisher Vectors.

This work was presented in IROS 2018 in Madrid, Spain and was also  published in Robotics and Automation Letters.

abstract: 

Modern robotic systems are often equipped with a direct 3D data acquisition device, e.g. LiDAR, which provides a rich 3D point cloud representation of the surroundings. This representation is commonly used for obstacle avoidance and mapping. Here, we propose a new approach for using point clouds for another critical robotic capability, semantic understanding of the environment (i.e. object classification). Convolutional neural networks (CNN), that perform extremely well for object classification in 2D images, are not easily extendible to 3D point clouds analysis. It is not straightforward due to point clouds' irregular format and a varying number of points. The common solution of transforming the point cloud data into a 3D voxel grid needs to address severe accuracy vs memory size tradeoffs. In this paper we propose a novel, intuitively interpretable, 3D point cloud representation called 3D Modified Fisher Vectors (3DmFV). Our representation is hybrid as it combines a coarse discrete grid structure with continuous generalized Fisher vectors. Using the grid enables us to design a new CNN architecture for real-time point cloud classification. In a series of performance analysis experiments, we demonstrate competitive results or even better than state-of-the-art on challenging benchmark datasets while maintaining robustness to various data corruptions.



Instructions: 
For training: 
1. Download The data directory from the onedrive folder in the link below. 

2. Run train.m  ( with desired parameters - number of Gaussians, number of points, augmentations, etc).
This will also create a log directory and sub-directories based on the number of points and Gaussians. 

For testing: 

1. 
1.a.Downloadthe log directory form the onedrive link below 
1.b. Alternatively, test your own trained model by training first.

2. Run test.m (make sure to set the path to the desired trained model).

link tp data and log: 
https://technionmail-my.sharepoint.com/:f:/g/personal/cadlab_technion_ac_il/EiylwxAQ4VxEpEo0Njqu59wBKohZwVRdG2LkaBoQCqvZ0w?e=4cwv1l


Citation: 
If you use this code please cite 

@article{ben20183dmfv,
  title={3DmFV: Three-Dimensional Point Cloud Classification in Real-Time Using Convolutional Neural Networks},
  author={Ben-Shabat, Yizhak and Lindenbaum, Michael and Fischer, Anath},
  journal={IEEE Robotics and Automation Letters},
  volume={3},
  number={4},
  pages={3145--3152},
  year={2018},
  publisher={IEEE}
}

It is trained on the ModelNet40 dataset by princeton, so please cite their work as well. 



