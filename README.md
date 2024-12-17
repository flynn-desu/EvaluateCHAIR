# Language

- [[中文]](README_zh.md)
- [[Engilish]](README.md)

# Environment Setup

1. **Create a Conda Environment (Skip if already exists)**

    ```sh
    conda create -n chair python==3.12
    ```

2. **Activate the Environment**

    ```sh
    conda activate chair
    ```

3. **Install Required Libraries**

    ```sh
    pip install nltk==3.9.1
    pip install textblob==0.18.0.post0
    ```

4. **Run `init.py`**

    ```python
    import nltk
    
    nltk.download('punkt_tab')
    ```

    - If you're using WSL and located in China, you might need to configure the proxy:

        ```python
        import nltk
        import os
        
        os.environ['http_proxy'] = 'http://10.255.34.44:7890'
        os.environ['https_proxy'] = 'http://10.255.34.44:7890'
        
        nltk.download('punkt_tab')
        ```


# Evaluating CHAIR Metrics

## Steps

1. **Determine your MSCOCO version** and place the captions and instance JSON files into the `coco_annotations` directory of the project (if you don't have them, you can download from the [official website](https://cocodataset.org/#download)).

    For example, my MSCOCO version is 2017, so my `coco_annotations` folder contains the following files:

    ```
    .
    ├── captions_train2017.json
    ├── captions_val2017.json
    ├── instances_train2017.json
    └── instances_val2017.json
    ```

2. **Organize your caption results** and place them in the `captions_to_eval` directory. The format of the JSON files needs to match the example I provided:

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

3. In `chair.py`, you can choose to evaluate a single file or an entire folder.

    ```python
    def eval_file_chair_example():
        eval_file_chair('./captions_to_eval/Clip.pth.json')
    
    
    def eval_dir_chair_example():
        eval_dir_chair()
    
    
    if __name__ == '__main__':
        eval_dir_chair_example()
    ```

    - Assume we are evaluating the entire folder (just run `chair.py`).

        - My `captions_to_eval` directory:

            ```sh
            captions_to_eval/
            ├── Clip.pth.json
            └── ViT.pth.json
            ```

        - Output printed by `chair.py` during the run:

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

        - In the `eval_result` directory, you will get `n` files, where `n` is the number of files in `captions_to_eval`, containing the evaluation results of CHAIR metrics:

            ```sh
            eval_result/
            ├── Clip.pth.json_eval_result.json
            └── ViT.pth.json_eval_result.json
            ```

        - In the `chair_compare` directory, you will find one more file that summarizes the CHAIR metrics for the caption results in the entire dataset:

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


## Customization

In `chair.py`, you can customize the input and output directories.

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

# References

- This project is a modified version of the [official code](https://github.com/LisaAnne/Hallucination), primarily relying on [chair.py](https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py).
- paper：[Object Hallucination in Image Captioning](https://arxiv.org/pdf/1809.02156)
