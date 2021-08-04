### 伪标签方式进行数据标注

1. 将标签数据放入dataset/tagged/目录下并命名为label.txt
2. 将未标记数据放入dataset/source/目录下并命名为unlabeled.txt
3. 运行run_psedo_labeling.sh

##### 注：默认为二分类，label.txt中的数据为"sentence\t1" or "sentence\t0"，如果想要更改为多分类，那么需要修改模型。运行算法之前需要在config.yaml中修改配置参数
