import torch
from transformers import MarianMTModel, MarianTokenizer

def get_valid_language_code(prompt):
    while True:
        lang_code = input(prompt).strip().lower()
        if lang_code in ["de", "en", "fr", "es"]:  # Add more valid language codes
            return lang_code
        print("Invalid language code. Please choose from available codes.")


def translate_text(input_text, source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True)
    translation_ids = model.generate(inputs['input_ids'])

    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translated_text


def translate_multiple_sentences(sentences, source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    inputs = tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True)
    translation_ids = model.generate(inputs['input_ids'], num_return_sequences=len(sentences))

    translated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in translation_ids]
    return translated_texts


if __name__ == "__main__":
    print("Welcome to the Language Translation App!")

    source_language = get_valid_language_code("Enter source language code (e.g., 'de' for German): ")
    target_language = get_valid_language_code("Enter target language code (e.g., 'en' for English): ")

    while True:
        print("\nSelect an option:")
        print("1. Translate a single sentence")
        print("2. Translate multiple sentences")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            input_text = input(f"Enter a sentence in {source_language}: ")
            translated_text = translate_text(input_text, source_language, target_language)
            print(f"Input text ({source_language}): {input_text}")
            print(f"Translated text ({target_language}): {translated_text}")

        elif choice == "2":
            num_sentences = int(input("How many sentences would you like to translate? "))
            input_sentences = [input(f"Enter sentence {i + 1}: ") for i in range(num_sentences)]
            translated_sentences = translate_multiple_sentences(input_sentences, source_language, target_language)

            print("\nTranslation Results:")
            for i, (input_sentence, translated_sentence) in enumerate(zip(input_sentences, translated_sentences)):
                print(f"{i + 1}. Input ({source_language}): {input_sentence}")
                print(f"   Translated ({target_language}): {translated_sentence}\n")

        elif choice == "3":
            print("Exiting the app. Goodbye!")
            break

        else:
            print("Invalid choice. Please select a valid option.")
