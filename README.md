# LLaMA_IF

模型文件存放在llama文件夹中，其中model.py包含的内容可以进行正常推理，直接使用python example.py就可以确认结果，另一个model_fixed_shape包含了固定了输入的attention block和transformer block，需要单独导出这两个模块的话，可以python export_fixed_module.py进行运行

需要注意的是如果要单独导出attention block的话，需要先初始化attention norm对input tensor进行一次预处理才能给attention block当作dummy input进行导出。

feed forward n同理。

2023.03.27 上传export transformer block v2代码，移除原有问题代码

目前试运行推理代码为example.py
导出特定模块代码为export_fp16_dynamic.py

上传export transformer block v3代码，对应为status的ipynb文件
v3版本对应的模型已经去除了内部动态算子的部分，并且量化精度测试完毕。

2023.03.28 上传了导出完整llama模型的代码，对应export full model的ipynb文件
目前只支持torch jit 和 torch script的格式导出，onnx版本目前仍然需要修复。


上传export 65B transformer block v1代码
完成了65B llama模型tfblock，token embedding 和last fc layer的导出代码。
对应文件为export 65B 的ipynb
