import os
import pandas as pd
import traceback
# parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class ResultRecorder:
    def __init__(self):
        self.my_df = self._create_dataframe_for_chat_result()

    def _create_dataframe_for_chat_result(self):
        column_names = ['Seq', 'HumanAnswer', 'GptAnswer', 'Model']
        df = pd.DataFrame(columns=column_names)
        return df

    def write_df_to_excel_file(self, model, debug_mode):
        destination_folder = os.path.join(parent_dir, 'results')
        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)
        # get the current time in the format of yyyymmdd_hhmm
        current_time = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        gpt_version = model.replace('-', '_').replace('.', '_')
        if debug_mode:
            output_excel_path = os.path.join(
                destination_folder, f'debug_{current_time}.xlsx')
        else:
            output_excel_path = os.path.join(
                destination_folder, f'{gpt_version}_{current_time}.xlsx')
        try:
            self.my_df.to_excel(output_excel_path,
                                index=False, engine='xlsxwriter')
        except:
            print(traceback.format_exc())
            # export and save my_df to csv as backup solution.
            print('Exported to csv instead.')
            self.my_df.to_csv(output_excel_path.replace(
                '.xlsx', '.csv'), index=False)

    def add_new_row_to_df(self, seq_info_batch, correct_answers_batch, recorded_conversation, model_returned):
        print(f'Recorded conversation is {recorded_conversation}')
        if type(recorded_conversation) is list:
            # that means this is full conversation. e.g. [{...},{...}]
            self.my_df.loc[self.my_df.shape[0]] = [
                seq_info_batch, correct_answers_batch,
                r'\n'.join([str(item) for item in recorded_conversation]),
                model_returned]
        else:
            # that means this is a single conversation. e.g. "last response from gpt."
            self.my_df.loc[self.my_df.shape[0]] = [
                seq_info_batch, correct_answers_batch,
                str(recorded_conversation),
                model_returned]
