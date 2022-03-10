# VFI_FMSMI
Video Frame Interpolation via Fusion of Multi-scale Motion Information

At present, a large number of deep learning methods have appeared in the existing video frame interpolation technology. The optical flow-based method requires three steps. First, the motion vector is estimated according to the optical flow, then the optical flow reverse map is established, and finally the interpolation operation is performed. The optical flow cannot handle the sudden change of brightness, object occlusion, and non-rigid deformation objects. The kernel-based method can complete motion estimation and motion compensation in one step, and has strong robustness.However, when the kernel size is smaller than the motion size of the object, the motion vector cannot be estimated, and increasing the kernel size will result in an overflow of computing power and cannot meet the practical requirements. 
To solve this problem, this paper proposes a Video Frame Interpolation via Fusion of Multi-scale Motion Information.
We use kernels to perform motion estimation and motion compensation at multiple scales, then interpolate to obtain intermediate frames of multiple scales, and finally fuse multiple intermediate frames and send them to the refinement network to obtain accurate intermediate frames. In addition, considering the possible occlusion of the front and rear frames, we simultaneously estimate the occlusion of the front and rear frames with the kernel, which further improves the accuracy of the intermediate frames.Experiments show that our method can solve the frame insertion error when the kernel size is smaller than the object motion size. Compared with other kernel methods, we have better results on high-resolution videos, and our method is in Middlebury and UCF101. on reaching the-state-of-art level.<br />


The model struction figure is as follows:
![image](https://github.com/lzylyx/VFI_FMSMI/blob/main/fig/model_struct.png)<br />

refineNet:
![image](https://github.com/lzylyx/VFI_FMSMI/blob/main/fig/refine.png)<br />

Versus results with other video interpolation method:
![image](https://github.com/lzylyx/VFI_FMSMI/blob/main/results/compare_results.png)<br />


Video compare result with SenseTime algorithm:
[![Watch the video](https://github.com/lzylyx/VFI_FMSMI/blob/main/video/left_sensetime_right_me.png)](https://github.com/lzylyx/VFI_FMSMI/blob/main/video/left_sensetime_right_me.mp4)


contact:

The code provided is for academic research purposes only. If your interested in this technology, please email with Lzy_Lyx@163.com