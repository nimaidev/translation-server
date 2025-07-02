from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import signal
import sys

import uvicorn

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None



def initialize_model_and_tokenizer(ckpt_dir, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=4,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


# en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"  # ai4bharat/indictrans2-en-indic-dist-200M
en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-dist-200M"

en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, quantization)

indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-dist-200M"

indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir, quantization)

ip = IndicProcessor(inference=True)


app = FastAPI()


class Translate(BaseModel):
    input_sentence : str
    source_lan : str
    target_lang: str


lang_list = [
        "eng_Latn", # Latin English
        "ben_Beng", # Bengali
        "pan_Guru", # Punjabi
        "asm_Beng", # Assamese
        "gom_Deva", # Konkani
        "guj_Gujr", # Gujarati
        "hin_Deva", # Hindi
        "kan_Knda", # Kannada,
        "mal_Mlym", # Malayalam
        "ory_Orya", # Odia,
        "tam_Taml", # Tamil,
        "tel_Telu", # Telugu
    ]

# Assamese (asm_Beng)	Kashmiri (Arabic) (kas_Arab)	Punjabi (pan_Guru)
# Bengali (ben_Beng)	Kashmiri (Devanagari) (kas_Deva)	Sanskrit (san_Deva)
# Bodo (brx_Deva)	Maithili (mai_Deva)	Santali (sat_Olck)
# Dogri (doi_Deva)	Malayalam (mal_Mlym)	Sindhi (Arabic) (snd_Arab)
# English (eng_Latn)	Marathi (mar_Deva)	Sindhi (Devanagari) (snd_Deva)
# Konkani (gom_Deva)	Manipuri (Bengali) (mni_Beng)	Tamil (tam_Taml)
# Gujarati (guj_Gujr)	Manipuri (Meitei) (mni_Mtei)	Telugu (tel_Telu)
# Hindi (hin_Deva)	Nepali (npi_Deva)	Urdu (urd_Arab)
# Kannada (kan_Knda)	Odia (ory_Orya)	
# post method to translate
@app.post("/language-server/translate")
def translate(input : Translate):
    if input.source_lan  not in lang_list or input.target_lang not in lang_list:
        return {
            "message" : "Not a valid dialect"
        }
    
    model = None
    tokenizer = None
    if input.target_lang == "eng_Latn":
        model = indic_en_model
        tokenizer = indic_en_tokenizer
    else:
        model = en_indic_model
        tokenizer = en_indic_tokenizer
    translation = batch_translate(
        [input.input_sentence],  # Note: batch_translate expects a list
        src_lang=input.source_lan,
        tgt_lang=input.target_lang,
        model=model, 
        tokenizer=tokenizer,
        ip=ip  # Don't forget to pass the ip parameter
    )
    return {"translation": translation[0]} 



# Signal handler for graceful shutdown
def handle_sigterm(signum, frame):
    print("Received SIGTERM signal. Cleaning up models and exiting...")
    
    # Delete models to free GPU memory
    global en_indic_tokenizer, en_indic_model, indic_en_tokenizer, indic_en_model
    del en_indic_tokenizer, en_indic_model
    del indic_en_tokenizer, indic_en_tokenizer  # noqa: F821
    
    torch.cuda.empty_cache()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGTERM, handle_sigterm)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)