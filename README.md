# Brainstorm-framework

## Usage

### Prepare the APPS & CodeContest dataset
TODO: 整理code contest数据集转换格式代码，

### Prepare your api-keys
Put your api-keys to ./data/api_keys/your_api_keys_folder_name/*.jsonl
For example, [data/api_keys/your_keys/0.jsonl](data/api_keys/your_keys/0.jsonl). The example file also give you a sight of the format of api-key we required.

TODO：把预处理api keys的代码摘出来。
### Customize your prompts
You can modify [configs/apps/baseline.py](configs/apps/baseline.py) & [configs/apps/simple.py](configs/apps/simple.py) which demonstrates how one-stage or two-stage prompts should be organized separately. 

TODO:无代码地修改？否则需要详细解释config文件、涉及数据集字段，不太合适。
### Modify the script and run
Take apps_simple as an example. Just run the scrips as follow:
```bash
bash ./scripts/apps_simple.sh
```
TODO: 需要重构turbo.py来无代码修改地支持新config

### Train your customized ranker
TODO:整理ranker train 代码，整合

