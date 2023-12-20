# realtime audio to image with a christmas theme using Stable Diffusion Turbo and Whsiper 
# by Ed Fries
# public domain

#requirements:
#pip install --quiet --upgrade diffusers transformers accelerate
# realtime whisper audio code modified from: https://github.com/davabase/whisper_real_time

import pygame, random, sys, torch
from diffusers import AutoPipelineForText2Image

bonusPrompt = "a christmas themed " # this string is prepended to the prompt sent to SD. Change to anything you want to give your pictures a consistent theme
X=512       # you can use a larger size but SD Turbo makes better images at 512x512 resolution
Y=512
whisperModel = "openai/whisper-large-v3" #choices=["tiny", "base", "small", "medium", "large"]
whisperDevice = 'cuda' #'cpu' or 'cuda'

big=False  #set big=True for 16gb graphics cards
if big:
    overscale=2 #adjust to fill your screen
    sdModel = "stabilityai/sdxl-turbo"
else:
    overscale=1.5
    sdModel = "stabilityai/sd-turbo"

def InitRender():
    global pipe, font, scrn, info
    pipe = AutoPipelineForText2Image.from_pretrained(sdModel, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe = pipe.to("cuda")
    font = pygame.font.Font('freesansbold.ttf', 12)
    info = pygame.display.Info()
    scrn = pygame.display.set_mode((info.current_w,info.current_h),pygame.FULLSCREEN)

def RenderImage(prompt):
    seed = random.randint(0, sys.maxsize)

    images = pipe(
        prompt = prompt,
        guidance_scale = 0.0,
        width = X,
        height= Y,
        num_inference_steps = 4,
        generator = torch.Generator("cuda").manual_seed(seed),
        ).images
    images[0].save("output.jpg")

def ShowImage(caption):
    newX = overscale*X
    newY = overscale*Y
    
    scrn.fill((0,0,0))

    pic = pygame.image.load("output.jpg")
    imp = pygame.transform.scale(pic, (newX, newY)).convert()
    scrn.blit(imp, ((info.current_w-newX)/2, (info.current_h-newY)/2))

    #draw caption
    text = font.render(caption, True, (255,255,255))
    textRect = text.get_rect()
    textRect.center = (info.current_w/2, info.current_h-textRect.h)
    scrn.blit(text, textRect)

    pygame.display.flip()

import datetime
from queue import Queue
import speech_recognition as sr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import whisper
from tempfile import NamedTemporaryFile
from datetime import timedelta
import io

def init_hear_text():
    global data_queue
    global source
    global temp_file
    global transcription
    global phrase_timeout
    global whisperPipe
    argsmodel = whisperModel
    argsnon_english = False
    argsenergy_threshold = 1000
    argsrecord_timeout = 5
    argsphrase_timeout = 3
    
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = argsenergy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    source = sr.Microphone(sample_rate=16000)
        
    # Load / Download model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(whisperModel, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(whisperDevice)
    processor = AutoProcessor.from_pretrained(whisperModel)
    whisperPipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch.float16,
        device=whisperDevice)

    record_timeout = argsrecord_timeout
    phrase_timeout = argsphrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

def hear_text():
    # The last time a recording was retreived from the queue.
    phrase_time = datetime.datetime.utcnow() #= None
    # Current raw audio bytes.
    last_sample = bytes()
    while True:
        try:
            now = datetime.datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                #result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                result = whisperPipe(temp_file)
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                    print(text)
                    return(text)
                else:
                    transcription[-1] = text
            pygame.event.pump()

        except KeyboardInterrupt:
            return("QUIT")
            break

pygame.init()
init_hear_text()
InitRender()
while True:
    prompt = ""
    print("listening...")
    while prompt == "":
        prompt = hear_text()
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                quit()
    if (prompt == "Thank you."): # Whisper likes to return this when it's quiet
        continue
    if (prompt == "Terminate." or prompt == "terminate"):
        quit()
    RenderImage(bonusPrompt+prompt)
    ShowImage(prompt)
