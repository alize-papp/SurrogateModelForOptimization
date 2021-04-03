from IPython.display import Audio, display
import numpy as np

def allDone(play_time_seconds=1):
    """Generate a noise. 
    Useful to notify when a piece of code that takes a long time is done computing. 

    Args:
        play_time_seconds ([float], optional): How many seconds the sound lasts. Defaults to 1:float.
    """
    framerate = 4410
    t = np.linspace(0, play_time_seconds, framerate*play_time_seconds)
    audio_data = np.sin(2*np.pi*300*t) + np.sin(2*np.pi*300*t)
    display(Audio(audio_data, rate=framerate, autoplay=True))


