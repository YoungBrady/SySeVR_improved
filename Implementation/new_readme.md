1. 在source2slice中执行slice_nvd.sh s 0 
    * s表示切片（也可以换成c表示计数，适用c可以查看切片结果，有多少成功，多少失败），0表示批次（一共划分了0-19，20个批次，可以使用多个docker进行并行切片）。
    * 源代码存放在../Program_data/file2。
    * 如果重开docker执行切片，请根据需要修改slice_nvd.sh中相关的路径。  
        *  docker pull tk1037/sagpool:1.0 下载镜像，创建镜像时将sysevr目录映射到/home/sysevr下即可。

    * 切片结果存放在../slice_all文件夹中。

2. 在source2slice中执行dealfile.py
    * 根据diff文件提取减号行作为切片标注的依据，结果存储在vulline_dict.json中

3. 在source2slice中执行make_label_nvd.py
    * 根据上一步提取的信息标注切片，结果存储../labels/labels.json中
4. 在source2slice中执行data_preprocess.py
    * 切片去重并以CVE为单位存放在../label_source路径下
5. 在data_preprocess中执行process_dataflow_func.py
    * 对切片分词，映射后存储在../corpus中
6. 在data_preprocess中执行create_w2vmodel.py
    * 训练词向量模型，存储在./w2v_model/wordmodel中
7. 在data_preprocess中执行dealcorpus.py
    * 对切片截断并去重
8. 在data_preprocess中执行get_dl_input.py
    * 转向量并拆分测试集和训练集
9. 在data_preprocess中执行dealrawdata.py
    * 将向量填充，统一长度
10. 