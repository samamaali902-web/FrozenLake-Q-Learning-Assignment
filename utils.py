import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]

def plot_progress(rewards, output="training_progress.png"):
    window = 200
    avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(window, len(rewards)+1), avg)
    plt.xlabel("Episode")
    plt.ylabel("Moving success rate (200 episodes)")
    plt.title("FrozenLake Q-learning training")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()

def save_gif(states, output="agent_demo.gif"):
    frames=[]
    palette={"S":"#cce5ff", "F":"#edf6fc", "H":"#383838", "G":"#a5e6af"}
    for state in states:
        im=Image.new("RGB",(400,430),"white"); d=ImageDraw.Draw(im)
        for r in range(4):
            for c in range(4):
                x,y=20+c*90,20+r*90
                tile=MAP[r][c]
                d.rectangle((x,y,x+88,y+88),fill=palette[tile],outline="#808080",width=2)
                d.text((x+38,y+35),tile,fill="black")
        r,c=divmod(state,4)
        x,y=20+c*90,20+r*90
        d.ellipse((x+23,y+23,x+65,y+65),fill="#ff8c42",outline="black",width=2)
        d.text((20,393),"Orange circle = trained agent",fill="black")
        frames.append(im)
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=420,loop=0)
