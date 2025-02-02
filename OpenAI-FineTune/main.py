from openai import OpenAI
import pandas as pd
import json
import random
from dotenv import load_dotenv
from instruct_few_shot import few_shot

load_dotenv()
client = OpenAI()


def load_json_data(file_path):
    try:
        df = pd.read_json(file_path)
        text = df['text'].to_list()
        category = df['type'].to_list()
        data_list = []
        for i in range(len(text)):
            system_dict = {"role": "system", "content": few_shot}
            user_dict = {"role": "user", "content": text[i]}
            assistant_dict = {"role": "assistant", "content": category[i]}

            messages_list = [system_dict, user_dict, assistant_dict]

            data_list.append({"messages": messages_list})
        return data_list
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None


# create train/validation split
def train_val_split(data_list):
    validation_index_list = random.sample(range(0, len(data_list) - 1), 9)
    validation_data_lst = [data_list[index] for index in validation_index_list]
    for val_data in validation_data_lst:
        data_list.remove(val_data)
    return data_list, validation_data_lst


# write examples to file
def create_training_json_file(train_data, validation_data):
    with open('prepared_data/training-data.jsonl', 'w') as train_file:
        for tr_data in train_data:
            json.dump(tr_data, train_file)
            train_file.write('\n')

    with open('prepared_data/validation-data.jsonl', 'w') as valid_file:
        for val_data in validation_data:
            json.dump(val_data, valid_file)
            valid_file.write('\n')


# Create training job
def create_train_job():
    training_file = client.files.create(file=open("prepared_data/training-data.jsonl", "rb"), purpose="fine-tune")
    validation_file = client.files.create(file=open("prepared_data/validation-data.jsonl", "rb"), purpose="fine-tune")
    client.fine_tuning.jobs.create(
        training_file=training_file.id,
        validation_file=validation_file.id,
        suffix="Doc-classify",
        model="gpt-3.5-turbo"
    )
    return client, training_file.id, validation_file.id


# Inference
def inference(text):
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0567:personal:doc-classify:3fUyWcel",
        messages=[
            {"role": "system", "content": few_shot},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0].message.content


if __name__ == "__main__":
    data = load_json_data('data/document.json')
    train_data_list, validation_data_list = train_val_split(data)
    create_training_json_file(train_data_list, validation_data_list)
    client, train_id, validation_id = create_train_job()

    test_data = ['Form 1040 U.S. Individual Income Tax Return 2023\nFiling Status: Single\nName: Oliver Brown\nSocial Security Number: 345-67-XXXX\nTotal Income: $102,800\nAdjusted Gross Income: $99,500\nTotal Tax: $21,588\nTotal Payments: $22,588\nRefund Amount: $1,000',
                 'U.S. PASSPORT\nType: P\nPassport No: 3344556677\nSurname: WILSON\nGiven Names: CHARLOTTE ANNE\nNationality: UNITED STATES OF AMERICA\nDate of Birth: 12 JUL 1988\nPlace of Birth: PHILADELPHIA, U.S.A\nDate of Issue: 01 MAR 2022\nDate of Expiration: 28 FEB 2032,'
                 'PAY STATEMENT\nCompany: AI Innovations Ltd.\nEmployee: Oliver Brown\nPay Period: 09/01/2024 - 09/15/2024\nGross Pay: $7,200.00\nFederal Tax: $1,440.00\nState Tax: $504.00\nSocial Security: $446.40\nMedicare: $104.40\nNet Pay: $4,705.20\nYTD Gross: $14,400.00',
                 'GLOBAL BANK\nMONTHLY STATEMENT\nAccount: ****2233\nStatement Period: 09/01/2024 - 09/30/2024\nBeginning Balance: $27,500.50\nDeposits/Credits: $15,200.00\nWithdrawals/Debits: $11,450.25\nEnding Balance: $31,250.25\nDirect Deposits: TECH SOLUTIONS $8,500.00',
                 '2023 W-2 Wage and Tax Statement\nEmployees social security number 345-67-XXXX\nEmployer identification number 34-5678901\nEmployee name: Oliver Brown\nWages, tips, other compensation: $102,800.00\nSocial security wages: $102,800.00\nMedicare wages and tips: $102,800.00\nFederal income tax withheld: $21,588.00']
    for data in test_data:
        inference(data)


