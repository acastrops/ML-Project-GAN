import os
from scipy.misc import imresize, imsave
import glob
import shutil
from matplotlib.image import imread
from PIL import Image



input_dir = './POKEMON_NEW3/'
input_path = os.path.join(input_dir,'*g')

input_files = glob.glob(input_path)

output_dir = './POKEMON_NEW2/'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

width = 100
height = 100
for i, fp in enumerate(input_files):
    # image = Image.open(fp)
    # image.convert("RGBA") # Convert this to RGBA if possible
    #
    # canvas = Image.new('RGBA', image.size, (255,255,255,255)) # Empty canvas colour (r,g,b,a)
    # canvas.paste(image, mask=image) # Paste the image onto the canvas, using it's alpha channel as mask
    # canvas.thumbnail([width, height], Image.ANTIALIAS)
    # canvas.save(output_dir + "/" + str(i) + ".png", format="PNG")
    img = imread(fp)
    img = img[:,:,:3]
    # img = imresize(img, (100, 100))
    imsave(output_dir + "/" + str(i) + ".png", img)
