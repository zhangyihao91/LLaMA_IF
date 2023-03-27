# LLaMA_IF

模型文件存放在llama文件夹中，其中model.py包含的内容可以进行正常推理，直接使用python example.py就可以确认结果，另一个model_fixed_shape包含了固定了输入的attention block和transformer block，需要单独导出这两个模块的话，可以python export_fixed_module.py进行运行

需要注意的是如果要单独导出attention block的话，需要先初始化attention norm对input tensor进行一次预处理才能给attention block当作dummy input进行导出。

feed forward n同理。

2023.03.27 上传export transformer block v2代码，移除原有问题代码
