#
# Use OPENAI API to make a Q&A program.
# This program is for data collection of my thesis.
# Original idea: Qin Li
# Create date: 2024-05-05
#

import os
import sys
import asyncio
import time
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from src.data_loader import question_and_answer_pair_batch_generator  # noqa
from src.gpt_coversation import run_a_predefined_conversation_chain  # noqa
from src.result_recorder import ResultRecorder  # noqa


def timer(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The function took {elapsed_time} seconds to run.")
        return result
    return wrapper


@timer
async def main(raw_input_path, manual_analysis_path, model, role_system_needed=True, debug_mode=True):
    """
    Compare the answers from human and gpt models.
    """
    batch_size = 1 if debug_mode else 5
    comparison_generator = question_and_answer_pair_batch_generator(
        raw_input_path, manual_analysis_path, batch_size=batch_size)

    my_result_recorder = ResultRecorder()

    batch_counter = 0
    for (seq_info_batch, prompts_batch) in comparison_generator:
        batch_counter += 1
        print(f'Working on batch {batch_counter}.')
        # create a task group to run the conversation chains in parallel.
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run_a_predefined_conversation_chain(role_system_needed=role_system_needed,
                                                                        predefined_prompts_of_a_user=[
                                                                            prompts_batch[i]],
                                                                        model=model, role_for_task='task2')) for i in range(len(prompts_batch))]
        for i, each_task in enumerate(tasks):
            last_response_from_gpt, model_returned, full_conversation = each_task.result()
            # add new row to the dataframe
            my_result_recorder.add_new_row_to_df(
                seq_info_batch[i], 'n/a', last_response_from_gpt, model_returned)

        if debug_mode and batch_counter >= 2:
            break

    my_result_recorder.write_df_to_excel_file(model, debug_mode)


if __name__ == '__main__':
    raw_input_excel_abs_path = os.path.join(
        dir_path, 'Raw_Input_Data_20240508.xlsx')
    manual_analysis_excel_abs_path = os.path.join(
        dir_path, 'Manual_Analysis.xlsx')

    model_1 = 'gpt-3.5-turbo-0125'
    model_2 = 'gpt-4-turbo-2024-04-09'
    model_3 = 'gpt-4o-2024-05-13'
    asyncio.run(
        main(raw_input_path=raw_input_excel_abs_path, manual_analysis_path=manual_analysis_excel_abs_path, model=model_3, debug_mode=False))
