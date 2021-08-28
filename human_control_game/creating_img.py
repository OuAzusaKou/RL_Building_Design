import numpy as np
from PIL import Image

obs = np.zeros((1,40,30))

obs = obs + 100

obs = np.squeeze(obs, 0)

new_map = Image.fromarray(obs.astype('uint8'), mode='L')

new_map.save('./img/mt'+'.gif')