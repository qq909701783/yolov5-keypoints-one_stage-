# yolov5-keypoints-one_stage-
该关键点检测为one_stage(非二阶段)

这是一个基础版本，代码没怎么优化，如有疑问，请与本人进行技术交流

这是个人一个比较有意思的工业问题解决思路(这里以ICDAR数据集为例)

首先来看问题：无法框全，如果扩大训练边框，则会框出别的物体

![image](https://github.com/qq909701783/yolov5-keypoints-one_stage-/blob/main/test.jpg)

加入keypoint分支后与边框坐标相互约束，产生新的box：

![image](https://github.com/qq909701783/yolov5-keypoints-one_stage-/blob/main/end.jpg)
