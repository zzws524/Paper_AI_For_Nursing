# Code For Paper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This open-sourced project is for the paper "Leveraging Artificial Intelligence to Uncover Key Contextual Factors in Complex Nursing Practices".

All the running results (feedback from GPT) are recorded in results folder.


## Contributors
Qinli (秦丽) is the major contributor of this paper. All the conceptual work and analysis was done by Qinli.

Ziwen helped with the code development. 

We appreciate all the authors from Nurseslabs.com. They provided the questions and human answers. 

## Tasks

- Task 1: Answer questions using chat gpt with different models.

- Task 2: Compare answers from human and LLM (large language model). List the contexts causing the different ideas.

## Requirements

- Python 3.11

## Installation

You can install the library using pip:

```bash
pip install -r requirements.txt
```

## Usage 
- For task 1, run
```bash
python task1_answer_question.py
```
- For Task 2, run
```bash
python task2_answer_comprison.py
```

- In task1_answer_question.py and task2_answer_comprison.py, you may change parameters in "asyncio.run(main())" at the end of the file.

    | Parameter | Type | Description | 
    | ----------- | ----------- | ----------- |
    | debug_mode | Bool | If it is set to true, the program only works on first few items. |
    | model | String | The name of GPT model. |
    | raw_input_path | String | The path of Raw_Input_Data excel file. |
    | manual_analysis_path | String | The path of Manual_Analysis excel file. |
    | role_system_needed | Bool | If it is set to true, the program will add 'role: system' for GPT. |
    
- Do NOT change the sheet name of each excel file. I may improve this limitation if anyone has the request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Basically you can use, modify and publish the code freely. 

## Contact
If you have any questions or suggestions, free free to open an issue or [Email Me](mailto:zzws524@sina.com)


