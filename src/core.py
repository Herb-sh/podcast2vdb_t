import os
import json
import ffmpeg
import whisperx
import requests
import torch
from torch import cuda
from dotenv import dotenv_values
config = dotenv_values("../.env")

# update for mps(OSX) if needed
DEFAULT_DEVICE = 'cuda' if cuda.is_available() else ('cpu' if torch.backends.mps.is_available() else 'cpu')
DEFAULT_MODEL = 'large-v2' if cuda.is_available() else ('tiny' if torch.backends.mps.is_available() else 'tiny')
DEFAULT_BATCH_SIZE = 16 if cuda.is_available() else (4 if torch.backends.mps.is_available() else 4)
DEFAULT_COMPUTE_TYPE = 'float16' if cuda.is_available() else ('int8' if torch.backends.mps.is_available() else 'int8')
 
DIR_TARGET = '../data/transcripts/'
os.makedirs(DIR_TARGET, exist_ok=True)


def parse_transcript_diarized(data, episode_id:str):
    combined_result = {}
    result = []
    current_speaker = data[0]['speaker'] if data[0]['speaker'] else 'unknown'
    current_phrase = ""
    current_start = data[0]['start']
    current_end = data[0]['end']
    for line in data:
        if 'speaker' not in line  or line['speaker'] == "":
            line['speaker'] = 'unknown'
        if line['speaker'] == current_speaker:
            words = [word['word'] for word in line['words']]
            current_phrase += " " + ' '.join(words)
            current_end = line['end']
        else:
            tmp_dict = {'speaker': current_speaker, 'start': current_start, 'end': current_end, 'text': current_phrase.strip(), 'episode': episode_id}
            result.append(tmp_dict)
            current_speaker = line['speaker']
            words = [word['word'] for word in line['words']]
            current_phrase = ' '.join(words)
            current_start = line['start']
            current_end = line['end']
    # Append last phrase
    tmp_dict = {'speaker': current_speaker, 'start': current_start, 'end': current_end, 'text': current_phrase.strip(), 'episode': episode_id}
    result.append(tmp_dict)
    combined_result['parsed_diarization'] = result
    return combined_result


def parse_raw_transcript(data:list, episode_id:str):
    combined_result = {}
    combined_result['raw_transcript'] = data
    result = []
    for ele in data:
        tmp_dict = {'speaker': 'unknown', 'start': ele['start'], 'end': ele['end'], 'text': ele['text'], 'episode': episode_id}
        result.append(tmp_dict)
    combined_result['parsed_transcript'] = result
    return combined_result


def transcribe(episode_id, filename:str, dir_target:str=DIR_TARGET, diarize:bool=False, model:str=DEFAULT_MODEL, device:str=DEFAULT_DEVICE, compute_type:str=DEFAULT_COMPUTE_TYPE, config:dict=config, language='de'):
    print('Processing episode ' + str(episode_id) + ' ...')
    print('Running on ' + device + ' with compute type ' + compute_type + ' ...')
    print('Loading model ' + model + ' ...')
    model = whisperx.load_model(model, device=device, compute_type=compute_type, language=language)
    print('Loading audio file ' + filename + ' ...')
    audio = whisperx.load_audio(filename)
    print(len(audio))
    print('Transcribing audio content ...')
    result_raw = model.transcribe(audio, batch_size=DEFAULT_BATCH_SIZE)
    print('Loading alignment model ...', result_raw)
    align_model, metadata = whisperx.load_align_model(language_code=result_raw['language'], device=device)
    print('Finished load alignment model...')
    result_aligned = whisperx.align(result_raw['segments'], align_model, metadata, audio, device, return_char_alignments=False)
    print('Finished whisperx alignment..')
    if diarize:
        ## Diarization
        print('Load diarization model ...')
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=config['HF_TOKEN'], device=device)
        print('Diarizing audio content ...')
        diarize_segments = diarize_model(filename)
        ## Assign word speakers
        print('Assigning word speakers ...')
        result_diarized = whisperx.assign_word_speakers(diarize_segments, result_aligned)
        print('Parsing diarized transcript ...')
        result_final = parse_transcript_diarized(result_diarized['segments'], episode_id)
    else:
        print('Skipping diarization, parsing aligned transcript ...')
        result_final = parse_raw_transcript(result_aligned['segments'], episode_id)

    
    filename_target = dir_target + '.'.join(filename.split('/')[-1].split('.')[0:-1] + ['json'])
    print('Storing result to ' + filename_target + ' ...')
    with open(filename_target, 'w') as file:
        json.dump(result_final, file)

    return (filename_target, result_final)
#%%
