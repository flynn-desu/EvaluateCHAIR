# 语言

- [[中文]](README_zh.md)
- [[Engilish]](README.md)

# 环境搭建

1. 创建 conda 环境（如果已有则跳过）

    ```sh
    conda create -n chair python==3.12
    ```

2. 激活环境

    ```sh
    conda activate chair
    ```

3. 安装所需库

    ```
    pip install nltk==3.9.1
    pip install textblob==0.18.0.post0
    ```

4. 运行 init.py

    ```python
    import nltk
    
    nltk.download('punkt_tab')
    ```

    - 如果使用 WSL 且在中国，可能需要配置

        ```python
        import nltk
        import os
        
        # 10.255.34.44 是宿主机 IP
        # 7890 是代理端口
        os.environ['http_proxy'] = 'http://10.255.34.44:7890'
        os.environ['https_proxy'] = 'http://10.255.34.44:7890'
        
        nltk.download('punkt_tab')
        ```

# 评估 CHAIR 指标

## 步骤

1. 确定你的 MSCOCO 版本，将 captions 和 instance 的 json 文件放入项目的 `coco_annotations` 目录中（如果没有，则前往[官网](https://cocodataset.org/#download)下载）

    例如我的 MSCOCO 版本是 2017，那么我的 `coco_annotations` 文件夹内容如下：

    ```
    .
    ├── captions_train2017.json
    ├── captions_val2017.json
    ├── instances_train2017.json
    └── instances_val2017.json
    ```

2. 组织你的 caption 结果并放到 `captions_to_eval` 目录中，json 文件的格式需要和我提供的示例一致：

    ```json
    [
        {
            "image_id": 384350,
            "caption": "a red and white plane on a runway."
        },
        {
            "image_id": 540414,
            "caption": "a group of people sitting around a table."
        },
        {
            "image_id": 383443,
            "caption": "a bathroom with a sink and a mirror."
        },
        {
            "image_id": 253433,
            "caption": "a stuffed teddy bear sitting on a bed."
        },
        {
            "image_id": 314709,
            "caption": "a woman riding skis down a snow covered slope."
        }
    ]
    ```

3. 在 `chair.py` 中可以选择评估单个文件或整个文件夹

    ```python
    def eval_file_chair_example():
        eval_file_chair('./captions_to_eval/Clip.pth.json')
    
    
    def eval_dir_chair_example():
        eval_dir_chair()
    
    
    if __name__ == '__main__':
        eval_dir_chair_example()
    ```

    - 假设我们评估整个文件（直接运行 `chair.py` 即可）

        - 我的 `captions_to_eval` 目录

            ```sh
            captions_to_eval/
            ├── Clip.pth.json
            └── ViT.pth.json
            ```

        - chair.py 运行过程中的打印内容

            ```sh
            ------------------------------
            now eval : Clip.pth.json [1/2]
            each caption chair analysis -->: 100%|██████████| 5/5 [00:00<00:00, 3792.32it/s]
            Clip.pth.json eval done
            ------------------------------
            
            
            ------------------------------
            now eval : ViT.pth.json [2/2]
            each caption chair analysis -->: 100%|██████████| 5/5 [00:00<00:00, 3541.89it/s]
            ViT.pth.json eval done
            ------------------------------
            ```

        - 在 `eval_result` 会多出 n 个文件：CHAIR 指标的评估结果，n 对应 `captions_to_eval` 中的文件数量

            ```sh
            eval_result/
            ├── Clip.pth.json_eval_result.json
            └── ViT.pth.json_eval_result.json
            ```

        - `chair_compare` 会多出一个文件：`captions_to_eval` 中 caption 结果在整个数据集上的 CHAIR 指标的汇总

            ```json
            {
                "Clip.pth.json": {
                    "CHAIRs": 0.2,
                    "CHAIRi": 0.125
                },
                "ViT.pth.json": {
                    "CHAIRs": 0.6,
                    "CHAIRi": 0.3333333333333333
                }
            }
            ```

## 自定义

在 chair.py 中可以自定义输入和输出目录

```python
def eval_file_chair(
    captions_to_eval_path, 
    dir_to_dump='eval_result', 
    coco_version='2017'
):
	...


def eval_dir_chair(
        dir_to_eval='captions_to_eval',
        dir_to_dump='eval_result',
        chair_compare_path='chair_compare/chair_compare.json',
        coco_version='2017'
):
    ...
```

# 引用

- 该项目由[官方代码](https://github.com/LisaAnne/Hallucination)修改而来，主要依赖 [chair.py](https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py)
- 论文：[Object Hallucination in Image Captioning](https://arxiv.org/pdf/1809.02156)