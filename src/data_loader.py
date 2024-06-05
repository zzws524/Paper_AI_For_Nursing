import pandas as pd
import re


def load_exams_data(excel_path, sheet_name='Raw Data'):
    """load exams data from an excel file to pandas dataframe.images are ignored.
    Remove image recognition question, gap filling question and ranking question for simplicity.
    """
    # read excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # Totally 5 out of 450 questions will be removed.
    df = df.drop(df[(df['Question Type']) ==
                 'Image recognition question'].index)
    df = df.drop(df[(df['Question Type']) == 'Gap filling question'].index)
    df = df.drop(df[(df['Question Type']) == 'Ranking question'].index)
    print(f'Now {df.shape[0]} rows were left.')
    # for each paragrah in the column 'Question', remove the blank lines.
    df['Question'] = df['Question'].apply(lambda x: '\n'.join(
        [line for line in x.split('\n') if line.strip() != '']))
    df['Explanation'] = df['Explanation'].apply(lambda x: '\n'.join(
        [line for line in x.split('\n') if line.strip() != '']))
    # for lines in paragrah of the column 'Question', trim the leading and trailing spaces.
    df['Question'] = df['Question'].apply(
        lambda x: '\n'.join([line.strip() for line in x.split('\n')]))
    df['Explanation'] = df['Explanation'].apply(
        lambda x: '\n'.join([line.strip() for line in x.split('\n')]))

    # Splitting question into stem and options is base on needs.
    # df = _split_question_into_stem_and_options(df)
    return df


def _split_question_into_stem_and_options(my_df):
    """split a question into stem and options.
    The 'question stem' and 'question options' are stored in new columns.
    """
    # search each 'Question'. Using regex to find the 'A.'.
    # before 'A.' is the question stem, after 'A.' is the options.
    my_df['Question_Stem'] = my_df['Question'].apply(
        lambda x: re.search(r'(.*)A\.', x, re.DOTALL).group(1))
    my_df['Question_Options'] = my_df['Question'].apply(
        lambda x: re.search(r'(A\..*)', x, re.DOTALL).group(1))
    return my_df


def questions_batch_generator(excel_path, sheet_name='Raw Data', batch_size=10):
    """make a batch generator for questions.
    """
    df = load_exams_data(excel_path, sheet_name=sheet_name)
    for i in range(0, df.shape[0], batch_size):
        batch_questions = []
        for j in range(i, min(i+batch_size, df.shape[0])):
            prompts_for_one_question = {
                'role': 'user', 'content': df.iloc[j]['Question']}
            batch_questions.append(prompts_for_one_question)
        # seq info batch, correct answers batch , prompts batch
        yield (list(df.iloc[i:min(i+batch_size, df.shape[0])]['Seq']), list(df.iloc[i:min(i+batch_size, df.shape[0])]['Result']), batch_questions)


def get_sequences_of_diff(excel_path, sheet_name='Summary', filter_col_name=r'Human VS 4o', filter_value='diff'):
    """Identify the sequence of questions where the answers differ between a human and GPT.
    """
    # read excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f'{sheet_name} has {df.shape[0]} rows.')
    # filter the rows by the value of the column 'Human VS 4o'.
    df = df[df[filter_col_name] == filter_value]
    # transform df['Seq'] as a list.
    sequences_of_diff_list = df['Seq'].tolist()
    print(
        f'GPT has {len(sequences_of_diff_list)} answers that are different from those of humans.')
    return sequences_of_diff_list


def load_gpt_data(excel_path, sheet_name='gpt_4o'):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f'{sheet_name} has {df.shape[0]} rows.')
    # for each paragrah in the column, remove the blank lines.
    df['GptAnswer'] = df['GptAnswer'].apply(lambda x: '\n'.join(
        [line for line in x.split('\n') if line.strip() != '']))
    # for lines in paragrah of the column, trim the leading and trailing spaces.
    df['GptAnswer'] = df['GptAnswer'].apply(
        lambda x: '\n'.join([line.strip() for line in x.split('\n')]))
    return df


def filter_answers_of_diff(df, sequences_of_diff_list):
    df = df[df['Seq'].isin(sequences_of_diff_list)]
    return df


def load_merged_masterlist_of_different_answers(exam_excel, gpt_excel, gpt_sheet='gpt_4o', filter_col='Human VS 4o'):
    exam_df = load_exams_data(exam_excel)
    gpt_df = load_gpt_data(gpt_excel)
    diff_list = get_sequences_of_diff(gpt_excel, 'Summary', filter_col, 'diff')
    exam_diff_df = filter_answers_of_diff(exam_df, diff_list)
    gpt_diff_df = filter_answers_of_diff(gpt_df, diff_list)
    assert exam_diff_df.shape[0] == gpt_diff_df.shape[0]
    # merge two dataframes
    merged_df = pd.merge(exam_diff_df, gpt_diff_df, on='Seq')
    return merged_df


def question_and_answer_pair_batch_generator(exam_excel, gpt_excel, batch_size=10):
    """make a batch generator for Question_HumanAnswer_GptAnswer.
    """
    df = load_merged_masterlist_of_different_answers(exam_excel, gpt_excel)
    for i in range(0, df.shape[0], batch_size):
        batch_Question_HumanAnswer_GptAnswer = []
        for j in range(i, min(i+batch_size, df.shape[0])):
            tmp_content = '###Question###\n' + \
                df.iloc[j]['Question'] + '\n' + '###Answer of Nurse_A###\n' + df.iloc[j]['Explanation'] + \
                '\n' + '###Answer of Nurse_B###\n' + \
                df.iloc[j]['GptAnswer'] + '\n' + \
                'Which nurse is correct?Why?If they are both correct, hightlight the key context causing the difference.'
            prompts_for_one_question = {
                'role': 'user', 'content': tmp_content}
            batch_Question_HumanAnswer_GptAnswer.append(
                prompts_for_one_question)
        # seq info batch, correct answers batch , prompts batch
        yield (list(df.iloc[i:min(i+batch_size, df.shape[0])]['Seq']), batch_Question_HumanAnswer_GptAnswer)


# debug
if __name__ == '__main__':
    tmp_yield = question_and_answer_pair_batch_generator(
        './Raw_Input_Data_20240508.xlsx', './Manual_Analysis.xlsx', batch_size=1)
    print(next(tmp_yield))
