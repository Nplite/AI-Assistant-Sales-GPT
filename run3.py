import argparse
import os
import json
import pyttsx3
import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatOpenAI
from dotenv import find_dotenv, load_dotenv

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

load_dotenv(find_dotenv())

engine = pyttsx3.init()


def collect_voice_data(num_samples):
    recognizer = sr.Recognizer()

    for i in range(num_samples):
        print(f"Collecting sample {i+1}")
        with sr.Microphone() as source:
            print("Speak now...")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            # Save the audio recording and its transcript to your dataset
            save_audio_recording(audio, f"sample_{i+1}.wav")
            save_transcript(text, f"sample_{i+1}.txt")
            print("Sample collected successfully!")
        except sr.UnknownValueError:
            print("Unable to recognize speech")
        except sr.RequestError as e:
            print(f"Error occurred during speech recognition: {e}")


def save_audio_recording(audio, filename):
    # Save the audio recording to a file
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())


def save_transcript(text, filename):
    # Save the transcript to a text file
    with open(filename, "w") as f:
        f.write(text)


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Unable to recognize speech")
    except sr.RequestError as e:
        print(f"Error occurred during speech recognition: {e}")
    return ""


def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()




def preprocess_transcript(transcript):
    # Remove unwanted characters or symbols
    transcript = re.sub(r"[^a-zA-Z0-9\s]", "", transcript)

    # Convert to lowercase
    transcript = transcript.lower()

    # Split into individual tokens
    tokens = transcript.split()

    # Remove stopwords or irrelevant words
    stopwords = ["the", "a", "an", "is", "are", "in", "on", "at"]
    tokens = [token for token in tokens if token not in stopwords]

    # Perform additional preprocessing as needed
    # ...

    # Join tokens back into a processed transcript
    processed_transcript = " ".join(tokens)

    return processed_transcript



def fine_tune_language_model(train_file, output_dir):
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt-3.5-turbo-0613"  # or any other pre-trained GPT variant
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load and preprocess the training data
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128  # Adjust block size based on your data and hardware limitations
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Fine-tuning configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,  # Adjust the number of training epochs based on your dataset size and resources
        per_device_train_batch_size=8,  # Adjust batch size based on your hardware limitations
        save_total_limit=2,
    )

    # Create Trainer instance and perform fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def generate_response(prompt, model_path):
    # Load fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate response
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response


def main():
    # Initialize argparse
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add arguments
    parser.add_argument('--config', type=str, help='Path to agent config file', default='')
    parser.add_argument('--verbose', type=bool, help='Verbosity', default=False)
    parser.add_argument('--max_num_turns', type=int, help='Maximum number of turns in the sales conversation', default=10)

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    config_path = args.config
    verbose = args.verbose
    max_num_turns = args.max_num_turns

    llm = ChatOpenAI(temperature=0.9, stop="<END_OF_TURN>")

    if config_path == '':
        print('No agent config specified, using a standard config')
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f'Agent config {config}')
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose, **config)

    sales_agent.seed_agent()
    print('-' * 20)
    cnt = 0
    while cnt != max_num_turns:
        cnt += 1
        if cnt == max_num_turns:
            print('Maximum number of turns reached - ending the conversation.')
            break
        sales_agent.step()

        # End conversation
        if '<END_OF_CALL>' in sales_agent.conversation_history[-1]:
            print('Sales Agent determined it is time to end the conversation.')
            break
        user_input = speech_to_text()
        sales_agent.human_step(user_input)
        print('-' * 20)


if __name__ == "__main__":
    main()
