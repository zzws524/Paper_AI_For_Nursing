import os
from openai import AsyncOpenAI
import pandas as pd
import textwrap

# Need config the OPENAI_API_KEY in the environment variable.
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


async def run_a_predefined_conversation_chain(role_system_needed, predefined_prompts_of_a_user, model, role_for_task='task1'):
    """A Predefined chain means the contents of 'user' are predfined, regardless the response of response from 'assistant'.
    We simply append the response from 'assistant' to the historical prompts.
    """
    _check_message_format(predefined_prompts_of_a_user)

    if role_system_needed:
        # The first role in historical_prompt is "system".
        if role_for_task == 'task1':
            historical_prompts = _init_role_system_prompt_for_task1_answer_question()
        elif role_for_task == 'task2':
            historical_prompts = _init_role_system_prompt_for_task2_answer_comparison()
        else:
            raise ValueError(
                'role_for_task must be either "task1" or "task2".')
    else:
        historical_prompts = []

    for one_round_prompts_from_user in predefined_prompts_of_a_user:
        historical_prompts.append(one_round_prompts_from_user)
        content_returned, model_returned = await _get_single_chat_completion(historical_prompts, model)
        print(f'{model_returned} now responded: \n{content_returned}')
        new_gpt_response = {"role": "assistant", "content": content_returned}
        historical_prompts.append(new_gpt_response)
    print('Finished one user/question.')
    # print(f'Full Conversation: {historical_prompts}')
    return (content_returned, model_returned, historical_prompts)


def _init_role_system_prompt_for_task1_answer_question():
    system_prompt = textwrap.dedent("""
        # ROLE #
        You are an experienced registered nurse in US.
        # OBJECTIVE #
        Based on the knowledge of nursing practice, select the most appropriate answers.
        - First provide your final answer as short as possible. In the format of "Correct Answer: A".
        - Then explain why the answer is selected. Analyze each option. 
        """).strip()
    return [{'role': 'system', 'content': system_prompt}]


def _init_role_system_prompt_for_task2_answer_comparison():
    system_prompt = textwrap.dedent("""
        # ROLE #
        You are an experienced registered nurse in US.
        # OBJECTIVE #
        Based on the knowledge of nursing practice, compare the answers from two nurses.
        - Select the right answer. In the format of 'Correct Answer: Nurse_A /Nurse_B /Both.Depends on context'.
        - Explain the reason. In the format of 'Reason: ...'.
        - List the contexts causing the different ideas. In the format of 'Different contextual considerations: ...'.
        """).strip()
    return [{'role': 'system', 'content': system_prompt}]


async def _get_single_chat_completion(historical_prompts, model):
    """single chat completion.

    Args:
        historical_prompts (dict): e.g. [{"role": "system", "content": "count 1"},{"role":"user","content": "count 2"}]
        model (str, optional): e.g. "gpt-3.5-turbo".

    Returns:
        tuple: (content_returned, model_returned).
    """
    _check_message_format(historical_prompts)
    chat_completion = await client.chat.completions.create(
        model=model,
        messages=historical_prompts,
    )
    if chat_completion.choices[0].finish_reason == "stop" or chat_completion.choices[0].message.content:
        model_returned = chat_completion.model
        content_returned = chat_completion.choices[0].message.content
        return (content_returned, model_returned)
    else:
        print(
            f'Error ! Content returned : {chat_completion.choices[0].finish_reason}')
        print(f'chat_completion : {chat_completion}')
        raise ValueError("Chat completion failed.")


def _check_message_format(historical_prompts):
    """check historical_prompts format for single chat completion.
    e.g. [{"role": "system", "content": "count 1"}, {"role": "user", "content": "count 2"}]
    """
    assert isinstance(historical_prompts, list)
    for each_round_prompts in historical_prompts:
        assert isinstance(each_round_prompts, dict)
        assert "role" in each_round_prompts and "content" in each_round_prompts
