# DoYouSeeWhatIHear 

## What is DoYouSeeWhatIHear?
DYSWIH is a simple python program that listens to your microphone, sends what it collects to Whisper to convert speech to text, adds a little Christmas magic, and then sends the modified text to Stable Diffusion to render an image. It's a fun program to leave running while you chat with friends or watch TV. It will pick up words and phrases and turn them into images to enhance your holiday enjoyment. Here's a short video of the program in action: https://youtu.be/jScgvOFkXmo

## Installation

DYSWIH runs great on my laptop with an 8gb Nvidia 3070. For anything smaller you'll likely have to use a smaller/older version of Whisper and run it on the cpu.

You'll want to have the usual torch/pytorch stuff installed. If you haven't used other local AI programs before, install something like https://github.com/AUTOMATIC1111/stable-diffusion-webui and once you have that running it should be easy to run this as well. 

After that, download or clone this git repo onto your local machine.

Make sure you have the needed dependencies by running pip:

```bash
pip install --quiet --upgrade diffusers transformers accelerate pygame
```

After that you should just be able run it from a command line with:

```python
python dyswih.py
```

It should create a fullscreen python window and start listening to your microphone. Have fun making Christmas images and just say "Terminate." when you want the program to stop.

                 -Ed Fries  12/20/2023
